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

# Import GPT-2 tokenizer
from transformers import GPT2Tokenizer

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Use FlashAttention if enabled
        if self.use_flash_attention:
            from FlashAttention import attention
            sm_scale = 1.0 / math.sqrt(self.head_dim)
            print("Q shape:", Q.shape, "K shape:", K.shape, "V shape:", V.shape)
            attn_output = attention(Q, K, V, True, sm_scale) # Assuming causal masking
        else:
            # Scaled dot-product attention
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)

        # Reshape and apply output projection
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
        # Multi-head self-attention with residual connection
        x = x + self.attn(self.ln1(x), attention_mask)
        # Feed-forward network with residual connection
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

        # Create position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.head(hidden_states)

        return logits

class BinaryDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.data_path = data_path
        self.seq_length = seq_length

        # Load the entire binary file into memory
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.vocab_size = 50257  # GPT-2 vocab size

        # Calculate the number of sequences
        self.num_sequences = (len(self.data) - 1) // self.seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 for the target

        # Get input and target sequences
        x = torch.tensor(self.data[start_idx:end_idx - 1], dtype=torch.long)
        y = torch.tensor(self.data[start_idx + 1:end_idx], dtype=torch.long)

        return x, y

class GPT2Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0  # Initialize start_epoch
        self.log_interval = 100

        # Determine global rank for distributed scenarios
        if dist.is_available() and dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = 0

        # Print device inside GPT2Trainer's __init__ method
        if dist.is_available() and dist.is_initialized():
            print(f"[Rank {self.global_rank}]: Inside GPT2Trainer __init__, assigned device: cuda:{torch.cuda.current_device()}")
        else:
            print(f"Inside GPT2Trainer __init__, using device: {device}")

        # Initialize GPT-2 model
        if self.args.run_type == 'baseline':
            self.model = GPT2Model(
                vocab_size=args.vocab_size,
                embedding_size=args.embedding_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_position_embeddings=args.max_position_embeddings,
                use_flash_attention=False
            )
            # Use DataParallel for baseline
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)

            self.model.to(device)
        else:
            # If not baseline, we initialize the model later with DeepSpeed
            self.model = GPT2Model(
                vocab_size=args.vocab_size,
                embedding_size=args.embedding_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_position_embeddings=args.max_position_embeddings,
                use_flash_attention=False   
            )

        # Initialize tokenizer for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Initialize dataset (only the Dataset, not DataLoader)
        train_data_path = args.train_data_path
        val_data_path = args.val_data_path

        self.train_dataset = BinaryDataset(train_data_path, args.seq_length)
        self.val_dataset = BinaryDataset(val_data_path, args.seq_length)

        if self.args.run_type == 'baseline':
            # Initialize DataLoader
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=args.train_micro_batch_size_per_gpu,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=args.train_micro_batch_size_per_gpu,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )

        else:
            # Load DeepSpeed configuration
            with open(args.deepspeed_config, 'r') as f:
                deepspeed_config = json.load(f)

            # Update DeepSpeed config for pipeline parallelism
            deepspeed_config['pipeline'] = {
                'stages': args.pipeline_stages,
                'partition_method': 'parameters',
                'seed_layers': True,
                'activation_checkpointing': {
                    'partition_activations': True,
                    'contiguous_memory_optimization': True,
                    'cpu_checkpointing': True
                }
            }

            # Initialize DeepSpeed
            self.model_engine, self.optimizer, self.train_dataloader, self.scheduler = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                training_data=self.train_dataset,
                config=deepspeed_config
            )

            # Initialize DeepSpeed DataLoader separately for validation
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=args.train_micro_batch_size_per_gpu,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=True
            )

        # Load checkpoint if resume is set
        if args.resume and os.path.exists(args.checkpoint_path):
            self.load_checkpoint(args.checkpoint_path)

        # Initialize metrics
        self.baseline_metrics = {
            "train_time_per_epoch": [],
            "train_loss_per_epoch": [],
            "val_loss_per_epoch": [],
            "inference_latency": [],
            "inference_throughput": [],
            "max_memory_usage": [],
            "seq_length_feasible": [args.seq_length],  # Store as a list
            "gpus_tested": [torch.cuda.device_count() if torch.cuda.is_available() else 0],
            "tokens_per_sec": []
        }

        self.optimized_metrics = copy.deepcopy(self.baseline_metrics)

    def load_checkpoint(self, checkpoint_path):
        # Load a checkpoint
        if self.args.run_type == 'baseline':
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            print(f"Loaded baseline checkpoint from {checkpoint_path}")
        else:
            load_path, _ = self.model_engine.load_checkpoint(checkpoint_path, tag=None)
            print(f"Loaded optimized checkpoint from {load_path}")

    def save_metrics(self, metrics, file_path):
        """
        Save metrics to a JSON file.

        Args:
            metrics (dict): Dictionary of metrics to save.
            file_path (str): Path to the JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def load_metrics(self, file_path):
        """
        Load metrics from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Dictionary of loaded metrics.
        """
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        return metrics

    def train(self):
        # Determine global rank for distributed scenarios
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
            # Print device at the beginning of the train() method
            print(f"[Rank {global_rank}]: Starting train() method, assigned device: cuda:{torch.cuda.current_device()}")
        else:
            global_rank = 0
            print(f"Starting train() method, using device: {device}")

        profile_path = os.path.join(self.args.checkpoint_path, f"{self.args.run_type}_profiler_logs")

        # Start the profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:

            # Store the start time of training
            self.train_start_time = time.time()

            for epoch in range(self.start_epoch, self.args.epochs):
                print(f"[Rank {global_rank}]: Starting Epoch {epoch + 1}/{self.args.epochs}")
                self.epoch_start_time = time.time()
                if self.args.run_type == 'baseline':
                    self.model.train()
                else:
                    self.model_engine.train()
                epoch_loss = 0.0
                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}", disable=(global_rank != 0))

                for step, batch in enumerate(progress_bar):
                    print(f"[Rank {global_rank}]: Starting Step {step} of Epoch {epoch + 1}")
                    input_ids, targets = batch
                    if self.args.run_type == 'baseline':
                        input_ids = input_ids.to(device)
                        targets = targets.to(device)
                        inputs = (input_ids, None)
                    else:
                        # For DeepSpeed, we assume one GPU per rank
                        local_device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else device
                        input_ids = input_ids.to(local_device)
                        targets = targets.to(local_device)
                        inputs = (input_ids, None)

                    self.optimizer.zero_grad()
                    print(f"[Rank {global_rank}]: Zeroed gradients")

                    try:
                        if self.args.run_type == 'baseline':
                            # Pass the tuple as *inputs
                            outputs = self.model(*inputs)
                        else:
                            outputs = self.model_engine(*inputs)
                        print(f"[Rank {global_rank}]: Completed forward pass")
                    except Exception as e:
                        print(f"[Rank {global_rank}]: Exception during forward pass: {e}")
                        raise e

                    try:
                        outputs = outputs.contiguous()
                        loss = F.cross_entropy(
                            outputs.view(-1, self.args.vocab_size), targets.view(-1)
                        )
                        print(f"[Rank {global_rank}]: Computed loss: {loss.item():.4f}")
                    except Exception as e:
                        print(f"[Rank {global_rank}]: Exception during loss computation: {e}")
                        raise e

                    # Reset peak memory stats after the first forward pass of the first epoch
                    if step == 0 and epoch == 0:
                        if self.args.run_type == 'baseline':
                            torch.cuda.reset_peak_memory_stats(device)
                            print(f"[Rank {global_rank}]: Reset peak memory stats for device {device}")
                        else:
                            # On optimized runs, we still reset on global_rank=0 GPU
                            if global_rank == 0:
                                torch.cuda.reset_peak_memory_stats(device)
                                print(f"[Rank {global_rank}]: Reset peak memory stats for device {device}")

                    try:
                        if self.args.run_type == 'baseline':
                            loss.backward()
                            self.optimizer.step()
                            print(f"[Rank {global_rank}]: Completed backward pass and optimizer step")
                        else:
                            self.model_engine.backward(loss)
                            self.model_engine.step()
                            print(f"[Rank {global_rank}]: Completed backward pass and optimizer step via DeepSpeed")
                    except Exception as e:
                        print(f"[Rank {global_rank}]: Exception during backward or optimizer step: {e}")
                        raise e

                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())

                    prof.step()

                    if step % self.log_interval == 0:
                        # Check memory stats only on global rank 0 to avoid clutter
                        if global_rank == 0:
                            memory_stats = torch.cuda.memory_stats(device)
                            if 'peak_bytes_allocated' in memory_stats:
                                peak_memory = memory_stats["peak_bytes_allocated"] / (1024 ** 2)  # Convert to MB
                            else:
                                peak_memory = 0.0
                            print(f"[Rank {global_rank}]: Step {step}, Training Loss: {loss.item():.4f}, Peak Memory: {peak_memory:.2f}MB")

                self.epoch_end_time = time.time()
                self.epoch_time = self.epoch_end_time - self.epoch_start_time
                avg_loss = epoch_loss / len(self.train_dataloader)
                print(f"[Rank {global_rank}]: Epoch {epoch+1}/{self.args.epochs}, Average Training Loss: {avg_loss:.4f}, Epoch Time: {self.epoch_time:.2f} seconds")

                # Adding print statements before and after validation
                print(f"[Rank {global_rank}]: Starting validation after Epoch {epoch + 1}")
                self.validate()
                print(f"[Rank {global_rank}]: Completed validation after Epoch {epoch + 1}")

                # Adding print statements before and after checkpoint saving
                print(f"[Rank {global_rank}]: Starting checkpoint saving after Epoch {epoch + 1}")
                if self.args.run_type == 'baseline':
                    if self.global_rank == 0:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                        checkpoint_path = os.path.join(self.args.checkpoint_path, f"baseline_epoch-{epoch+1}.pt")
                        torch.save(checkpoint, checkpoint_path)
                        print(f"[Rank {global_rank}]: Baseline checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
                elif self.args.run_type == 'optimized':
                    
                    checkpoint = self.model_engine.save_checkpoint(self.args.checkpoint_path, tag=f"epoch-{epoch+1}")
                    print(f"[Rank {global_rank}]: Optimized checkpoint saved at epoch {epoch+1} to {checkpoint}")
                print(f"[Rank {global_rank}]: Completed checkpoint saving after Epoch {epoch + 1}")

                # Adding barrier to ensure all ranks have completed the epoch before proceeding
                if dist.is_available() and dist.is_initialized():
                    print(f"[Rank {global_rank}]: Waiting at barrier after Epoch {epoch + 1}")
                    dist.barrier()
                    print(f"[Rank {global_rank}]: Passed barrier after Epoch {epoch + 1}")

                # Continue to next epoch
                print(f"[Rank {global_rank}]: Moving to the next epoch")

            # Store the end time of training
            self.train_end_time = time.time()
            self.total_training_time = self.train_end_time - self.train_start_time
            print(f"[Rank {global_rank}]: Total training time: {self.total_training_time:.2f} seconds")

            if self.args.run_type == 'baseline':
                if global_rank == 0:
                    if 'peak_bytes_allocated' in memory_stats:
                        self.baseline_metrics.setdefault("max_memory_usage", []).append(torch.cuda.max_memory_allocated(device) / (1024**2))
                    else:
                        self.baseline_metrics.setdefault("max_memory_usage", []).append(0.0)
                    print("Baseline metrics:", self.baseline_metrics)

            else:
                # Only collect metrics on rank 0 for optimized runs
                if global_rank == 0:
                    if 'peak_bytes_allocated' in memory_stats:
                        self.optimized_metrics.setdefault("max_memory_usage", []).append(torch.cuda.max_memory_allocated(device) / (1024**2))
                    else:
                        self.optimized_metrics.setdefault("max_memory_usage", []).append(0.0)
                    print("Optimized metrics:", self.optimized_metrics)

    def validate(self):
        if self.args.run_type == 'baseline':
            self.model.eval()
        else:
            self.model_engine.eval()
        
        print(f"[Rank {self.global_rank}]: Starting validation")
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                print(f"[Rank {self.global_rank}]: Validation batch {batch_idx}")
                input_ids, targets = batch
                if self.args.run_type == 'baseline':
                    input_ids = input_ids.to(device)
                    targets = targets.to(device)
                else:
                    input_ids = input_ids.to(self.model_engine.local_rank)
                    targets = targets.to(self.model_engine.local_rank)

                try:
                    # Forward pass
                    if self.args.run_type == 'baseline':
                        outputs = self.model(input_ids, attention_mask=None)
                    else:
                        outputs = self.model_engine(input_ids, attention_mask=None)
                    print(f"[Rank {self.global_rank}]: Completed forward pass during validation")
                except Exception as e:
                    print(f"[Rank {self.global_rank}]: Exception during validation forward pass: {e}")
                    raise e
                
                try:
                    loss = F.cross_entropy(outputs.view(-1, self.args.vocab_size), targets.view(-1))
                    print(f"[Rank {self.global_rank}]: Computed validation loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"[Rank {self.global_rank}]: Exception during validation loss computation: {e}")
                    raise e

                total_loss += loss.item()
                print(f"[Rank {self.global_rank}]: Validation batch {batch_idx} loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.val_dataloader)
        print(f"[Rank {self.global_rank}]: Validation Loss: {avg_loss:.4f}")

        if self.args.run_type == 'baseline':
            self.baseline_metrics["val_loss_per_epoch"].append(avg_loss)
        else:
            self.optimized_metrics["val_loss_per_epoch"].append(avg_loss)
        print(f"[Rank {self.global_rank}]: Completed validation with Average Loss: {avg_loss:.4f}")

    def generate_text(self, prompt, max_length=50):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
        else:
            global_rank = 0

        # Only the global rank 0 process should perform text generation
        if global_rank != 0:
            print("Skipping text generation on non-zero rank.")
            return ""

        if self.args.run_type == 'baseline':
            self.model.eval()
            local_device = device
            model_to_use = self.model
        else:
            self.model_engine.eval()
            local_device = device
            model_to_use = self.model_engine

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(local_device)

        generated = input_ids
        self.generate_start_time = time.time()
        with torch.no_grad():
            for _ in range(max_length):
                if self.args.run_type == 'baseline':
                    # DataParallel support: inputs is a tuple
                    inputs = (generated, None)
                    outputs = model_to_use(*inputs)
                else:
                    outputs = model_to_use(generated, attention_mask=None)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token_id], dim=-1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        self.generate_end_time = time.time()
        self.generate_time = self.generate_end_time - self.generate_start_time
        generated_text = self.tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
        num_generated_tokens = generated.shape[1] - input_ids.shape[1]
        self.throughput = num_generated_tokens / self.generate_time

        print(f"Inference Latency: {self.generate_time:.4f} seconds")
        print(f"Inference Throughput: {self.throughput:.2f} tokens/second")

        if self.args.run_type == 'baseline':
            self.baseline_metrics.setdefault("inference_latency", []).append(self.generate_time)
            self.baseline_metrics.setdefault("inference_throughput", []).append(self.throughput)
            print("Baseline metrics after inference:", self.baseline_metrics)
        else:
            self.optimized_metrics.setdefault("inference_latency", []).append(self.generate_time)
            self.optimized_metrics.setdefault("inference_throughput", []).append(self.throughput)
            print("Optimized metrics after inference:", self.optimized_metrics)

        return generated_text

    def summarize_results(self):
        import torch.distributed as dist

        # Determine the global rank
        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
        else:
            global_rank = 0

        # Only rank 0 should print the summary
        if global_rank != 0:
            return

        def get_last_or_default(metric_dict, key, default_value="N/A"):
            """
            Retrieve the last value from a metric list or return a default value.
            """
            val = metric_dict.get(key, default_value)
            if isinstance(val, list):
                return val[-1] if val else default_value
            else:
                return val

        def calculate_improvement(baseline, optimized):
            """
            Calculate the percentage improvement between baseline and optimized metrics.
            Returns "N/A" if either value is not numeric or if baseline is zero.
            """
            if baseline == "N/A" or optimized == "N/A":
                return "N/A"
            try:
                baseline = float(baseline)
                optimized = float(optimized)
            except (ValueError, TypeError):
                return "N/A"
            if baseline == 0:
                return "N/A" if optimized == 0 else "Inf improvement"
            return ((baseline - optimized) / baseline) * 100

        def format_improvement(improvement):
            """
            Format the improvement value for display.
            """
            if isinstance(improvement, (int, float)):
                return f"{improvement:.2f}%"
            else:
                return improvement

        # Fetch the latest metrics
        baseline_train_time = get_last_or_default(self.baseline_metrics, 'train_time_per_epoch')
        optimized_train_time = get_last_or_default(self.optimized_metrics, 'train_time_per_epoch')
        train_time_reduction = calculate_improvement(baseline_train_time, optimized_train_time)

        baseline_inference_latency = get_last_or_default(self.baseline_metrics, 'inference_latency')
        optimized_inference_latency = get_last_or_default(self.optimized_metrics, 'inference_latency')
        inference_latency_improvement = calculate_improvement(baseline_inference_latency, optimized_inference_latency)

        baseline_inference_throughput = get_last_or_default(self.baseline_metrics, 'inference_throughput')
        optimized_inference_throughput = get_last_or_default(self.optimized_metrics, 'inference_throughput')
        inference_throughput_improvement = calculate_improvement(baseline_inference_throughput, optimized_inference_throughput)

        baseline_max_memory = get_last_or_default(self.baseline_metrics, 'max_memory_usage')
        optimized_max_memory = get_last_or_default(self.optimized_metrics, 'max_memory_usage')

        baseline_seq_length = get_last_or_default(self.baseline_metrics, 'seq_length_feasible')
        optimized_seq_length = get_last_or_default(self.optimized_metrics, 'seq_length_feasible')
        seq_length_reduction = calculate_improvement(baseline_seq_length, optimized_seq_length)

        baseline_gpus = get_last_or_default(self.baseline_metrics, 'gpus_tested')
        optimized_gpus = get_last_or_default(self.optimized_metrics, 'gpus_tested')
        gpus_reduction = calculate_improvement(baseline_gpus, optimized_gpus)

        baseline_train_throughput = get_last_or_default(self.baseline_metrics, 'train_throughput')
        optimized_train_throughput = get_last_or_default(self.optimized_metrics, 'train_throughput')
        train_throughput_improvement = calculate_improvement(baseline_train_throughput, optimized_train_throughput)

        # Print Summary Table
        print("===== Summary of Main Results =====")
        print("| Metric | Baseline | Optimized | Improvement |")
        print("|----------------------|--------------|---------------|----------------------|")
        print(f"| Training time per epoch (s) | {baseline_train_time} | {optimized_train_time} | {format_improvement(train_time_reduction)} |")
        print(f"| Inference latency (s) | {baseline_inference_latency} | {optimized_inference_latency} | {format_improvement(inference_latency_improvement)} |")
        print(f"| Inference throughput (tokens/s) | {baseline_inference_throughput} | {optimized_inference_throughput} | {format_improvement(inference_throughput_improvement)} |")
        print(f"| Max memory usage (MB) | {baseline_max_memory} | {optimized_max_memory} | N/A |")
        print(f"| Feasible sequence length | {baseline_seq_length:.2f} | {optimized_seq_length:.2f} | {format_improvement(seq_length_reduction)}% reduction |")
        print(f"| GPUs tested          | {baseline_gpus:.2f} | {optimized_gpus:.2f} | {format_improvement(gpus_reduction)}% reduction |")
        print(f"| Training throughput (tokens/s) | {baseline_train_throughput} | {optimized_train_throughput} | {format_improvement(train_throughput_improvement)} |")

        # Print Performance Improvements
        print("\nPerformance Improvements:")

        # Training Time Reduction
        print(f" - Training time reduced by: {format_improvement(train_time_reduction)} per epoch")

        # Inference Latency Improvement
        print(f" - Inference latency reduced by: {format_improvement(inference_latency_improvement)}")

        # Inference Throughput Improvement
        print(f" - Inference throughput improved by: {format_improvement(inference_throughput_improvement)}")

        # Training Throughput Improvement
        print(f" - Training throughput improved by: {format_improvement(train_throughput_improvement)}")

    def print_kernel_comparison(self):
        import glob

        # Find the trace files in the baseline and optimized profiler logs directories
        baseline_trace_files = glob.glob(os.path.join("checkpoint", "baseline", "baseline_profiler_logs", "*.pt.trace.json"))
        optimized_trace_files = glob.glob(os.path.join("checkpoint", "optimized", "optimized_profiler_logs", "*.pt.trace.json"))

        # Check if any baseline trace files were found
        if not baseline_trace_files:
            print("Warning: No baseline trace files found. Skipping kernel-level comparison.")
            return

        # Check if any optimized trace files were found
        if not optimized_trace_files:
            print("Warning: No optimized trace files found. Skipping kernel-level comparison.")
            return
        
        baseline_trace_file = baseline_trace_files[0]
        optimized_trace_file = optimized_trace_files[0]

        def process_trace(trace_file):
            with open(trace_file, 'r') as f:
                data = json.load(f)

            kernel_times = {}
            for event in data['traceEvents']:
                if 'cat' in event and event['cat'] == 'kernel':
                    kernel_name = event['name']
                    duration = event['dur'] / 1000.0  # Convert to milliseconds
                    kernel_times[kernel_name] = kernel_times.get(kernel_name, 0) + duration

            # Get top 3 kernels
            sorted_kernels = sorted(kernel_times.items(), key=lambda item: item[1], reverse=True)[:3]
            return dict(sorted_kernels)

        baseline_kernels = process_trace(baseline_trace_file)
        optimized_kernels = process_trace(optimized_trace_file)

        all_kernels = set(baseline_kernels.keys()).union(optimized_kernels.keys())

        print("===== Kernel-Level Analysis =====")
        print("| Kernel Name | Baseline Time (ms) | Optimized Time (ms) |")
        print("|---------------------------|--------------------|---------------------|")

        for kernel_name in all_kernels:
            baseline_time = baseline_kernels.get(kernel_name, 0)
            optimized_time = optimized_kernels.get(kernel_name, 0)
            print(f"| {kernel_name:<25} | {baseline_time:18.2f} | {optimized_time:19.2f} |")

        print(
            "\nThese results show the time spent in each kernel. A significant reduction in time for optimized kernels indicates successful optimization.")

def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 Model with DeepSpeed Parallelism')

    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--embedding_size', type=int, default=768, help='Embedding size')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_position_embeddings', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length for training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=8, help='Micro batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping norm')

    # Checkpoint parameters
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='Path to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    # Data paths
    parser.add_argument('--train_data_path', type=str, default='train.bin', help='Path to the training binary file')
    parser.add_argument('--val_data_path', type=str, default='val.bin', help='Path to the validation binary file')

    # DeepSpeed configuration
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json', help='Path to DeepSpeed config file')

    # Add DeepSpeed launcher arguments
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    # Run type: baseline or optimized
    parser.add_argument('--run_type', type=str, choices=['baseline', 'optimized'], required=True, help='Type of run: baseline or optimized')

    # Pipeline parallelism stages (for optimized)
    parser.add_argument('--pipeline_stages', type=int, default=2, help='Number of pipeline stages for pipeline parallelism')

    args = parser.parse_args()

    # Initialize process group for older PyTorch versions
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        torch.cuda.set_device(local_rank)

        # Initialize process group without device_ids
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=global_rank,
            world_size=world_size
        )
        print(f"[Rank {global_rank}]: Process started, assigned device: cuda:{torch.cuda.current_device()}")
        # Use a regular barrier (without device_ids)
        dist.barrier()  # Barrier with device_ids
    else:
        # Handle single-GPU case or non-distributed training
        global_rank = 0
        print(f"[Rank {global_rank}]: Running on a single GPU or in non-distributed mode.")

    # Only create directories on rank 0
    if global_rank == 0:
        if args.run_type == 'baseline':
            args.checkpoint_path = os.path.join(args.checkpoint_path, "baseline")
        else:
            args.checkpoint_path = os.path.join(args.checkpoint_path, "optimized")
        os.makedirs(args.checkpoint_path, exist_ok=True)

    trainer = GPT2Trainer(args)

    trainer.train()

    # Generate text and summarize results only on rank 0
    if global_rank == 0:
        prompt = "Once upon a time"
        generated_text = trainer.generate_text(prompt, max_length=50)
        print("Generated Text:")
        print(generated_text)

        # Compare results if running optimized
        if args.run_type == 'optimized':
            trainer.summarize_results()
            trainer.print_kernel_comparison()

if __name__ == '__main__':
    main()
