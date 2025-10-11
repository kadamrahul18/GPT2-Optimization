import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import argparse
import deepspeed
import json
import time
import torch.profiler
import copy
import torch.distributed as dist
import tiktoken

from transformers import GPT2Tokenizer

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, input_ids):
        return self.embedding(input_ids)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_size):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)

    def forward(self, position_ids):
        return self.position_embeddings(position_ids)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, use_flash_attention=False):
        super(MultiHeadSelfAttention, self).__init__()
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.use_flash_attention = use_flash_attention
        self.embedding_size = embedding_size

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash_attention:
            try:
                from FlashAttention import attention
                sm_scale = 1.0 / math.sqrt(self.head_dim)
                attn_output = attention(Q.contiguous(), K.contiguous(), V.contiguous(), True, sm_scale)
            except ImportError:
                 raise ImportError("FlashAttention not found or installed correctly. Cannot use use_flash_attention=True.")
        else:
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.fc2 = nn.Linear(4 * embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, use_flash_attention):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_size, eps=1e-5)
        self.attn = MultiHeadSelfAttention(embedding_size, num_heads, dropout, use_flash_attention)
        self.ln2 = nn.LayerNorm(embedding_size, eps=1e-5)
        self.ffn = FeedForward(embedding_size, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, dropout, max_position_embeddings, use_flash_attention=False):
        super(GPT2Model, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_size)
        self.position_embedding = PositionalEmbedding(max_position_embeddings, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.max_position_embeddings = max_position_embeddings
        self.use_flash_attention = use_flash_attention

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, num_heads, dropout, use_flash_attention) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embedding_size, eps=1e-5)
        self.head = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        if attention_mask is None and not self.use_flash_attention:
             mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device))
             attention_mask = mask.unsqueeze(0).unsqueeze(0)
             attention_mask = attention_mask.to(dtype=hidden_states.dtype)
             attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)

        logits = self.head(hidden_states)

        return logits

class BinaryDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.data_path = data_path
        self.seq_length = seq_length

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')

        self.num_sequences = (len(self.data) - 1) // self.seq_length
        if self.num_sequences == 0 and len(self.data) > self.seq_length:
             self.num_sequences = 1
        elif len(self.data) <= self.seq_length :
             print(f"Warning: Data length ({len(self.data)}) is less than or equal to sequence length ({self.seq_length}). Setting num_sequences to 0.")
             self.num_sequences = 0


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1

        actual_end_idx = min(end_idx, len(self.data))
        actual_x_end = min(start_idx + self.seq_length, len(self.data))

        x_data = self.data[start_idx : actual_x_end]
        y_data = self.data[start_idx + 1 : actual_end_idx]

        x_len = len(x_data)
        y_len = len(y_data)

        x = torch.from_numpy(x_data.copy()).to(torch.long)
        y = torch.from_numpy(y_data.copy()).to(torch.long)

        if x_len < self.seq_length:
            padding_x = torch.zeros(self.seq_length - x_len, dtype=torch.long)
            x = torch.cat((x, padding_x), dim=0)

        if y_len < self.seq_length:
             padding_y = torch.zeros(self.seq_length - y_len, dtype=torch.long)
             y = torch.cat((y, padding_y), dim=0)


        if len(x) != self.seq_length or len(y) != self.seq_length:
             print(f"Warning: Mismatch in sequence length at index {idx}. x: {len(x)}, y: {len(y)}, target: {self.seq_length}")
             if len(x) < self.seq_length: x = F.pad(x, (0, self.seq_length - len(x)))
             if len(y) < self.seq_length: y = F.pad(y, (0, self.seq_length - len(y)))
             x = x[:self.seq_length]
             y = y[:self.seq_length]


        return x, y

class GPT2Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.log_interval = args.steps_per_print

        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1
        if dist.is_available() and dist.is_initialized():
            self.global_rank = dist.get_rank()
            self.local_rank = args.local_rank
            self.world_size = dist.get_world_size()
            self.current_device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.current_device)
        else:
             self.current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        def rank_print(*print_args, **kwargs):
            if self.global_rank == 0:
                print(f"[Rank {self.global_rank}]", *print_args, **kwargs)

        self.rank_print = rank_print

        self.rank_print(f"Initializing GPT2Trainer on Global Rank {self.global_rank}, Local Rank {self.local_rank}, World Size {self.world_size}, Device {self.current_device}")

        self.rank_print("Initializing GPT-2 Model on CPU...")
        self.model = GPT2Model(
            vocab_size=args.vocab_size,
            embedding_size=args.embedding_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_position_embeddings=args.max_position_embeddings,
            use_flash_attention=args.use_flash_attention if hasattr(args, 'use_flash_attention') else False
        )
        self.rank_print("GPT-2 Model Initialized.")

        self.rank_print("Loading Tiktoken GPT-2 Tokenizer...")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.rank_print("Tokenizer Loaded.")
        self.tokenizer.eos_token_id = self.tokenizer.eot_token


        self.rank_print(f"Loading datasets: Train='{args.train_data_path}', Val='{args.val_data_path}'")
        train_data_path = args.train_data_path
        val_data_path = args.val_data_path
        try:
            self.train_dataset = BinaryDataset(train_data_path, args.seq_length)
            self.val_dataset = BinaryDataset(val_data_path, args.seq_length)
            self.rank_print(f"Datasets loaded: Train size={len(self.train_dataset)}, Val size={len(self.val_dataset)}")
            if len(self.train_dataset) == 0:
                 self.rank_print("ERROR: Training dataset has zero length. Check data path and preprocessing.")
                 sys.exit(1)
        except Exception as e:
            self.rank_print(f"ERROR: Failed to load datasets: {e}")
            sys.exit(1)


        try:
            with open(args.deepspeed_config, 'r') as f:
                deepspeed_config = json.load(f)
            self.rank_print(f"Loaded DeepSpeed config from {args.deepspeed_config}")
        except Exception as e:
             self.rank_print(f"ERROR: Failed to load DeepSpeed config: {e}")
             sys.exit(1)

        self.rank_print("Initializing DeepSpeed engine...")
        try:
             self.model_engine, self.optimizer, self.train_dataloader, self.scheduler = deepspeed.initialize(
                 model=self.model,
                 model_parameters=self.model.parameters(),
                 training_data=self.train_dataset,
                 config=deepspeed_config
             )
             self.rank_print("DeepSpeed Engine Initialized Successfully.")
             self.rank_print(f"Using device: {self.model_engine.local_rank} (mapped to {self.current_device})")


        except Exception as e:
             self.rank_print(f"ERROR: DeepSpeed Initialization Failed: {e}")
             if hasattr(e, 'extra_info'):
                 self.rank_print(f"Extra Info: {e.extra_info}")
             sys.exit(1)

        if dist.is_available() and dist.is_initialized():
             self.rank_print("Waiting at barrier after DeepSpeed initialization...")
             dist.barrier()
             self.rank_print("Passed barrier after DeepSpeed initialization.")

        if len(self.val_dataset) > 0:
            val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False)
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=args.train_micro_batch_size_per_gpu,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True,
                drop_last=True
            )
            self.rank_print(f"Validation DataLoader created. Size: {len(self.val_dataloader)}")
        else:
             self.rank_print("Validation dataset is empty, skipping validation dataloader creation.")
             self.val_dataloader = None


        if args.resume:
             self.rank_print(f"Attempting to resume from checkpoint path: {args.checkpoint_path}")
             pass

        self.metrics = {
            "train_time_per_epoch": [],
            "train_loss_per_epoch": [],
            "val_loss_per_epoch": [],
            "train_throughput": []
        }

        self.rank_print(f"GPT2Trainer initialization complete for Rank {self.global_rank}.")

    def train(self):
        self.rank_print(f"Starting train() method on device {self.current_device}")

        prof = None
        if self.global_rank == 0 and self.args.profile:
            profile_path = os.path.join(self.args.checkpoint_path, "profiler_logs")
            os.makedirs(profile_path, exist_ok=True)
            self.rank_print(f"Profiler enabled. Logs will be saved to {profile_path}")
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
                record_shapes=False,
                with_stack=False,
                profile_memory=True,
                with_flops=False
            )
            prof.__enter__()

        if dist.is_available() and dist.is_initialized():
            self.rank_print("Waiting at barrier before training loop...")
            dist.barrier()
            self.rank_print("Passed barrier, entering training loop.")

        if self.global_rank == 0:
            self.train_start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            self.rank_print(f"Starting Epoch {epoch + 1}/{self.args.epochs}")
            if self.global_rank == 0:
                self.epoch_start_time = time.time()

            self.model_engine.train()

            if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                 self.train_dataloader.sampler.set_epoch(epoch)
                 self.rank_print(f"Set dataloader sampler epoch to {epoch}")


            progress_bar = None
            if self.global_rank == 0:
                 progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.args.epochs} Rank 0")


            total_train_loss_epoch = 0.0
            num_tokens_processed_epoch = 0
            num_steps_epoch = 0

            if dist.is_available() and dist.is_initialized():
                 self.rank_print(f"Waiting at barrier: Start of Epoch {epoch+1} loop...")
                 dist.barrier()
                 self.rank_print(f"Passed barrier: Start of Epoch {epoch+1} loop.")


            for step, batch in enumerate(self.train_dataloader):
                 if dist.is_available() and dist.is_initialized():
                      if step % 50 == 0:
                           print(f"[Rank {self.global_rank}] Waiting at barrier: Start of Step {step}, Epoch {epoch+1}")
                      dist.barrier()
                      if step % 50 == 0:
                           print(f"[Rank {self.global_rank}] Passed barrier: Start of Step {step}, Epoch {epoch+1}")


                 input_ids, targets = batch
                 input_ids = input_ids.to(self.current_device)
                 targets = targets.to(self.current_device)
                 inputs = (input_ids, None)

                 try:
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Starting forward pass, Step {step}, Epoch {epoch+1}, Input Shape: {input_ids.shape}")
                      outputs = self.model_engine(*inputs)
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Completed forward pass, Step {step}, Epoch {epoch+1}, Output Shape: {outputs.shape}")
                 except Exception as e:
                      print(f"[Rank {self.global_rank}]: CRITICAL ERROR during forward pass: Step {step}, Epoch {epoch+1}: {e}")
                      import traceback
                      traceback.print_exc()
                      if dist.is_available() and dist.is_initialized(): dist.barrier()
                      sys.exit(1)

                 try:
                      outputs = outputs.contiguous()
                      loss = F.cross_entropy(
                          outputs.view(-1, self.args.vocab_size), targets.view(-1)
                      )
                      batch_loss = loss.item()
                      if step % 50 == 0 or step < 5:
                          print(f"[Rank {self.global_rank}]: Computed loss: {batch_loss:.4f}, Step {step}, Epoch {epoch+1}")

                 except Exception as e:
                      print(f"[Rank {self.global_rank}]: CRITICAL ERROR during loss computation: Step {step}, Epoch {epoch+1}: {e}")
                      import traceback
                      traceback.print_exc()
                      if dist.is_available() and dist.is_initialized(): dist.barrier()
                      sys.exit(1)

                 try:
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Starting backward pass, Step {step}, Epoch {epoch+1}")
                      self.model_engine.backward(loss)
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Completed backward pass, Step {step}, Epoch {epoch+1}")

                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Starting optimizer step, Step {step}, Epoch {epoch+1}")
                      self.model_engine.step()
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Completed optimizer step, Step {step}, Epoch {epoch+1}")

                 except Exception as e:
                      print(f"[Rank {self.global_rank}]: CRITICAL ERROR during backward/step: Step {step}, Epoch {epoch+1}: {e}")
                      import traceback
                      traceback.print_exc()
                      if dist.is_available() and dist.is_initialized(): dist.barrier()
                      sys.exit(1)


                 if self.global_rank == 0:
                      total_train_loss_epoch += batch_loss
                      num_tokens_processed_epoch += input_ids.numel()
                      num_steps_epoch += 1

                      if progress_bar:
                          progress_bar.update(1)
                          progress_bar.set_postfix(loss=batch_loss)

                      if step % self.log_interval == 0:
                           try:
                                memory_stats = torch.cuda.memory_stats(self.current_device)
                                peak_memory_mb = memory_stats.get("allocated_bytes.all.peak", 0) / (1024**2)
                                current_memory_mb = memory_stats.get("allocated_bytes.all.current", 0) / (1024**2)
                                self.rank_print(f"Step {step}, Loss: {batch_loss:.4f}, Mem Curr: {current_memory_mb:.2f}MB, Mem Peak: {peak_memory_mb:.2f}MB")
                                if step == 0 and epoch == 0:
                                      torch.cuda.reset_peak_memory_stats(self.current_device)
                                      self.rank_print("Reset peak memory stats after first step.")
                           except Exception as mem_e:
                                self.rank_print(f"Warning: Could not get memory stats: {mem_e}")

                 if prof:
                      prof.step()

            if self.global_rank == 0:
                if progress_bar:
                    progress_bar.close()
                self.epoch_end_time = time.time()
                self.epoch_time = self.epoch_end_time - self.epoch_start_time

                avg_loss_epoch = total_train_loss_epoch / num_steps_epoch if num_steps_epoch > 0 else 0
                approx_total_tokens = num_tokens_processed_epoch * self.world_size
                train_throughput = approx_total_tokens / self.epoch_time if self.epoch_time > 0 else 0

                self.rank_print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss_epoch:.4f}, Time: {self.epoch_time:.2f}s, Approx Throughput: {train_throughput:.2f} tokens/sec")

                self.metrics["train_time_per_epoch"].append(self.epoch_time)
                self.metrics["train_loss_per_epoch"].append(avg_loss_epoch)
                self.metrics["train_throughput"].append(train_throughput)

            if dist.is_available() and dist.is_initialized():
                self.rank_print(f"Waiting at barrier before validation, Epoch {epoch+1}...")
                dist.barrier()
                self.rank_print(f"Passed barrier, starting validation, Epoch {epoch+1}.")

            if self.val_dataloader:
                self.validate(epoch)
            else:
                self.rank_print("Skipping validation as dataloader is not available.")


            if dist.is_available() and dist.is_initialized():
                self.rank_print(f"Waiting at barrier before checkpointing, Epoch {epoch+1}...")
                dist.barrier()
                self.rank_print(f"Passed barrier, proceeding with checkpointing, Epoch {epoch+1}.")

            if self.global_rank == 0:
                try:
                     tag = f"epoch-{epoch+1}"
                     self.rank_print(f"Attempting to save checkpoint with tag '{tag}' to {self.args.checkpoint_path}...")
                     self.model_engine.save_16bit_model(self.args.checkpoint_path, tag)
                     self.rank_print(f"Checkpoint saved successfully with tag '{tag}'")
                except Exception as ckpt_e:
                     self.rank_print(f"ERROR: Failed to save checkpoint for epoch {epoch+1}: {ckpt_e}")


            if dist.is_available() and dist.is_initialized():
                self.rank_print(f"Waiting at barrier: End of Epoch {epoch+1} processing...")
                dist.barrier()
                self.rank_print(f"Passed barrier: End of Epoch {epoch+1} processing.")


        if self.global_rank == 0:
            self.train_end_time = time.time()
            self.total_training_time = self.train_end_time - self.train_start_time
            self.rank_print(f"Total training time: {self.total_training_time:.2f} seconds")

            if prof:
                 prof.__exit__(None, None, None)
                 self.rank_print("Profiler stopped.")

            metrics_file_path = os.path.join(self.args.checkpoint_path, "training_metrics.json")
            self.save_metrics(self.metrics, metrics_file_path)
            self.rank_print(f"Final metrics saved to {metrics_file_path}")

        if dist.is_available() and dist.is_initialized():
            self.rank_print("Waiting at final barrier...")
            dist.barrier()
            self.rank_print("Passed final barrier. Training finished.")


    def validate(self, epoch):
        self.rank_print(f"Starting validation for Epoch {epoch+1}")
        self.model_engine.eval()

        total_loss = 0.0
        total_steps = 0

        val_progress_bar = None
        if self.global_rank == 0:
             val_progress_bar = tqdm(total=len(self.val_dataloader), desc=f"Validation Epoch {epoch+1} Rank 0")

        with torch.no_grad():
            for step, batch in enumerate(self.val_dataloader):

                 input_ids, targets = batch
                 input_ids = input_ids.to(self.current_device)
                 targets = targets.to(self.current_device)
                 inputs = (input_ids, None)

                 try:
                      if step % 20 == 0 or step < 3:
                          print(f"[Rank {self.global_rank}]: Validation Step {step}, Epoch {epoch+1}, Input Shape: {input_ids.shape}")
                      outputs = self.model_engine(*inputs)
                      if step % 20 == 0 or step < 3:
                          print(f"[Rank {self.global_rank}]: Validation Step {step}, Epoch {epoch+1}, Output Shape: {outputs.shape}")

                      loss = F.cross_entropy(outputs.view(-1, self.args.vocab_size), targets.view(-1))
                      batch_loss = loss.item()
                      if step % 20 == 0 or step < 3:
                           print(f"[Rank {self.global_rank}]: Validation Loss: {batch_loss:.4f}, Step {step}, Epoch {epoch+1}")

                      if self.global_rank == 0:
                           total_loss += batch_loss
                           total_steps += 1
                           if val_progress_bar:
                               val_progress_bar.update(1)
                               val_progress_bar.set_postfix(loss=batch_loss)

                 except Exception as e:
                      print(f"[Rank {self.global_rank}]: CRITICAL ERROR during validation: Step {step}, Epoch {epoch+1}: {e}")
                      import traceback
                      traceback.print_exc()
                      if dist.is_available() and dist.is_initialized(): dist.barrier()
                      break

        if self.global_rank == 0:
            if val_progress_bar:
                val_progress_bar.close()

            avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
            self.rank_print(f"Validation Finished Epoch {epoch+1}. Average Loss: {avg_loss:.4f}")

            self.metrics["val_loss_per_epoch"].append(avg_loss)
        else:
             if "val_loss_per_epoch" in self.metrics:
                 pass

        if dist.is_available() and dist.is_initialized():
             self.rank_print(f"Waiting at barrier after validation, Epoch {epoch+1}...")
             dist.barrier()
             self.rank_print(f"Passed barrier after validation, Epoch {epoch+1}.")


    def generate_text(self, prompt, max_length=50):
        if self.global_rank != 0:
            if dist.is_available() and dist.is_initialized():
                 dist.barrier()
            return None

        self.rank_print(f"Starting text generation with prompt: '{prompt}'")
        self.model_engine.eval()

        try:
            prompt_ids = self.tokenizer.encode_ordinary(prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.current_device)
            self.rank_print(f"Input IDs shape: {input_ids.shape}, Device: {input_ids.device}")

            generated_ids_list = list(prompt_ids)
            current_generated_tensor = input_ids

            generate_start_time = time.time()
            with torch.no_grad():
                for i in range(max_length):
                    inputs = (current_generated_tensor, None)
                    outputs = self.model_engine(*inputs)
                    next_token_logits = outputs[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()

                    generated_ids_list.append(next_token_id)
                    current_generated_tensor = torch.tensor([generated_ids_list], dtype=torch.long).to(self.current_device)
                    

                    if next_token_id == self.tokenizer.eos_token_id:
                        self.rank_print(f"EOS token reached at step {i+1}.")
                        break

            generate_end_time = time.time()
            generate_time = generate_end_time - generate_start_time

            generated_text = self.tokenizer.decode(generated_ids_list)
            num_generated_tokens = len(generated_ids_list) - len(prompt_ids)
            throughput = num_generated_tokens / generate_time if generate_time > 0 else 0

            self.rank_print(f"Generated Text: {generated_text}")
            self.rank_print(f"Inference Latency: {generate_time:.4f} seconds")
            self.rank_print(f"Inference Throughput: {throughput:.2f} tokens/second")

        except Exception as e:
             self.rank_print(f"ERROR during text generation: {e}")
             import traceback
             traceback.print_exc()
             generated_text = f"Error during generation: {e}"

        if dist.is_available() and dist.is_initialized():
             self.rank_print("Waiting at barrier after generation...")
             dist.barrier()

        return generated_text

    def save_metrics(self, metrics, file_path):
        if self.global_rank == 0:
            self.rank_print(f"Saving metrics to {file_path}")
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
            except Exception as e:
                 self.rank_print(f"ERROR: Failed to save metrics to {file_path}: {e}")


    def load_metrics(self, file_path):
        if self.global_rank == 0:
            self.rank_print(f"Loading metrics from {file_path}")
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        metrics = json.load(f)
                    return metrics
                else:
                    self.rank_print(f"Warning: Metrics file not found at {file_path}")
                    return {}
            except Exception as e:
                 self.rank_print(f"ERROR: Failed to load metrics from {file_path}: {e}")
                 return {}
        else:
             return {}


    def summarize_results(self):
        if self.global_rank != 0:
            return

        self.rank_print("Summarizing results (comparison assumes baseline metrics are available).")
        optimized_metrics_path = os.path.join(self.args.checkpoint_path, "training_metrics.json")
        self.optimized_metrics = self.load_metrics(optimized_metrics_path)

        baseline_metrics_path = os.path.join(os.path.dirname(self.args.checkpoint_path), "baseline", "baseline_metrics.json")
        self.baseline_metrics = self.load_metrics(baseline_metrics_path)

        if not self.baseline_metrics or not self.optimized_metrics:
             self.rank_print("Cannot summarize results: Baseline or Optimized metrics missing.")
             return

        def get_last_or_default(metric_dict, key, default_value="N/A"):
            val = metric_dict.get(key, [])
            if isinstance(val, list):
                 return val[-1] if val else default_value
            else:
                 return val if val is not None else default_value

        def calculate_improvement(baseline, optimized, higher_is_better=False):
            if baseline == "N/A" or optimized == "N/A": return "N/A"
            try:
                baseline = float(baseline); optimized = float(optimized)
            except (ValueError, TypeError): return "N/A"
            if higher_is_better:
                if optimized == 0: return "N/A" if baseline == 0 else "Inf improvement"
                return ((optimized - baseline) / baseline) * 100 if baseline != 0 else "Inf improvement"
            else:
                if baseline == 0: return "N/A" if optimized == 0 else "Inf improvement"
                return ((baseline - optimized) / baseline) * 100

        def format_improvement(improvement):
            if isinstance(improvement, (int, float)): return f"{improvement:.2f}%"
            else: return improvement

        baseline_train_time = get_last_or_default(self.baseline_metrics, 'train_time_per_epoch')
        optimized_train_time = get_last_or_default(self.optimized_metrics, 'train_time_per_epoch')
        train_time_reduction = calculate_improvement(baseline_train_time, optimized_train_time, higher_is_better=False)

        baseline_inference_latency = get_last_or_default(self.baseline_metrics, 'inference_latency')
        optimized_inference_latency = get_last_or_default(self.optimized_metrics, 'inference_latency')
        inference_latency_improvement = calculate_improvement(baseline_inference_latency, optimized_inference_latency, higher_is_better=False)

        baseline_inference_throughput = get_last_or_default(self.baseline_metrics, 'inference_throughput')
        optimized_inference_throughput = get_last_or_default(self.optimized_metrics, 'inference_throughput')
        inference_throughput_improvement = calculate_improvement(baseline_inference_throughput, optimized_inference_throughput, higher_is_better=True)


        baseline_max_memory = get_last_or_default(self.baseline_metrics, 'max_memory_usage')
        optimized_max_memory = get_last_or_default(self.optimized_metrics, 'max_memory_usage')
        memory_reduction = calculate_improvement(baseline_max_memory, optimized_max_memory, higher_is_better=False)


        baseline_val_loss = get_last_or_default(self.baseline_metrics, 'val_loss_per_epoch')
        optimized_val_loss = get_last_or_default(self.optimized_metrics, 'val_loss_per_epoch')
        loss_improvement = calculate_improvement(baseline_val_loss, optimized_val_loss, higher_is_better=False)


        baseline_train_throughput = get_last_or_default(self.baseline_metrics, 'train_throughput')
        optimized_train_throughput = get_last_or_default(self.optimized_metrics, 'train_throughput')
        train_throughput_improvement = calculate_improvement(baseline_train_throughput, optimized_train_throughput, higher_is_better=True)


        print("\n===== Summary of Main Results =====")
        print("| Metric                          | Baseline     | Optimized    | Improvement       |")
        print("|---------------------------------|--------------|--------------|-------------------|")
        print(f"| Train Time/Epoch (s)          | {baseline_train_time: <12} | {optimized_train_time: <12} | {format_improvement(train_time_reduction): <17} |")
        print(f"| Validation Loss                 | {baseline_val_loss: <12} | {optimized_val_loss: <12} | {format_improvement(loss_improvement): <17} |")
        print(f"| Train Throughput (tokens/s)   | {baseline_train_throughput: <12} | {optimized_train_throughput: <12} | {format_improvement(train_throughput_improvement): <17} |")
        print(f"| Inference Latency (s)         | {baseline_inference_latency: <12} | {optimized_inference_latency: <12} | {format_improvement(inference_latency_improvement): <17} |")
        print(f"| Inference Throughput (tokens/s) | {baseline_inference_throughput: <12} | {optimized_inference_throughput: <12} | {format_improvement(inference_throughput_improvement): <17} |")
        print(f"| Max Memory Usage (MB)         | {baseline_max_memory: <12} | {optimized_max_memory: <12} | {format_improvement(memory_reduction): <17} |")


    def print_kernel_comparison(self):
        if self.global_rank != 0:
             return
        self.rank_print("Attempting kernel-level comparison...")
        import glob

        baseline_log_dir = os.path.join(os.path.dirname(self.args.checkpoint_path), "baseline", "baseline_profiler_logs")
        optimized_log_dir = os.path.join(self.args.checkpoint_path, "profiler_logs")

        baseline_trace_files = glob.glob(os.path.join(baseline_log_dir, "*.pt.trace.json"))
        optimized_trace_files = glob.glob(os.path.join(optimized_log_dir, "*.pt.trace.json"))

        if not baseline_trace_files:
            print("Warning: No baseline trace files found. Skipping kernel-level comparison.")
            return
        if not optimized_trace_files:
            print("Warning: No optimized trace files found. Skipping kernel-level comparison.")
            return

        baseline_trace_file = baseline_trace_files[0]
        optimized_trace_file = optimized_trace_files[0]

        def process_trace(trace_file):
            try:
                with open(trace_file, 'r') as f: data = json.load(f)
            except Exception as e:
                print(f"Error reading trace file {trace_file}: {e}")
                return {}

            kernel_times = {}
            for event in data.get('traceEvents', []):
                if event.get('cat') == 'kernel':
                    kernel_name = event.get('name', 'UnknownKernel')
                    duration = event.get('dur', 0) / 1000.0
                    kernel_times[kernel_name] = kernel_times.get(kernel_name, 0) + duration
            sorted_kernels = sorted(kernel_times.items(), key=lambda item: item[1], reverse=True)[:3]
            return dict(sorted_kernels)


        baseline_kernels = process_trace(baseline_trace_file)
        optimized_kernels = process_trace(optimized_trace_file)

        if not baseline_kernels or not optimized_kernels:
             print("Could not process kernel data from trace files.")
             return

        all_kernels = set(baseline_kernels.keys()).union(optimized_kernels.keys())

        print("===== Top 3 Kernel-Level Analysis =====")
        print("| Kernel Name                | Baseline Time (ms) | Optimized Time (ms) |")
        print("|----------------------------|--------------------|---------------------|")
        for kernel_name in all_kernels:
            baseline_time = baseline_kernels.get(kernel_name, 0)
            optimized_time = optimized_kernels.get(kernel_name, 0)
            display_name = (kernel_name[:25] + '...') if len(kernel_name) > 28 else kernel_name
            print(f"| {display_name:<26} | {baseline_time:18.2f} | {optimized_time:19.2f} |")

        print("\nNote: Shows total time for top kernels found in either trace. Lower time in optimized is better.")


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 Model with DeepSpeed')

    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--embedding_size', type=int, default=768, help='Embedding size (e.g., 768 for GPT-2 small)')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers (e.g., 12 for GPT-2 small)')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads (e.g., 12 for GPT-2 small)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_position_embeddings', type=int, default=1024, help='Maximum sequence length model can handle')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length for training data chunks')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use FlashAttention implementation')


    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=4, help='Micro batch size per GPU (overridden by deepspeed config)')
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (overridden by deepspeed config)')
    # parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate (can be overridden by deepspeed config scheduler)')
    # parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    # parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping norm (overridden by deepspeed config)')
    parser.add_argument('--steps_per_print', type=int, default=10, help='Frequency of printing training logs')

    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/optimized', help='Base path to save checkpoints and logs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint (handled by DeepSpeed)')

    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training binary file (train.bin)')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to the validation binary file (val.bin)')

    parser.add_argument('--deepspeed_config', type=str, required=True, help='Path to DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by DeepSpeed launcher')

    parser.add_argument('--run_type', type=str, choices=['baseline', 'optimized'], required=True, help='Type of run')

    # parser.add_argument('--pipeline_stages', type=int, default=1, help='Number of pipeline stages (overridden by deepspeed config)')

    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler (logs saved in checkpoint_path/profiler_logs)')


    args = parser.parse_args()


    if args.local_rank == -1:
         print("Warning: local_rank not set by launcher, running in non-distributed mode.")
         pass
    else:
         pass

    trainer = GPT2Trainer(args)

    trainer.train()

    if trainer.global_rank == 0:
        prompt = "To be or not to be, that is the question:"
        generated_text = trainer.generate_text(prompt, max_length=100)
        print("Generated Text:")
        print(generated_text)

        trainer.summarize_results()
        trainer.print_kernel_comparison()

if __name__ == '__main__':
    main()
