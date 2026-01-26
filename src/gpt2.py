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
import json
import time
from datetime import datetime, timezone
import torch.profiler
import copy
import torch.distributed as dist
import tiktoken
import metrics as metrics_utils

from transformers import GPT2Tokenizer

def get_dist_info(args):
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        local_rank = int(env_local_rank)
    else:
        arg_local_rank = getattr(args, "local_rank", None)
        if arg_local_rank is not None and int(arg_local_rank) >= 0:
            local_rank = int(arg_local_rank)
        else:
            local_rank = 0

    env_rank = os.environ.get("RANK")
    rank = int(env_rank) if env_rank is not None else 0

    env_world_size = os.environ.get("WORLD_SIZE")
    world_size = int(env_world_size) if env_world_size is not None else 1
    return local_rank, rank, world_size


def resolve_batch_config(args, ds_config, visible_gpus):
    micro_batch = (
        args.micro_batch_size_per_gpu
        if args.micro_batch_size_per_gpu is not None
        else ds_config.get("train_micro_batch_size_per_gpu", args.train_micro_batch_size_per_gpu)
    )
    grad_accum = (
        args.grad_accum_steps
        if args.grad_accum_steps is not None
        else ds_config.get("gradient_accumulation_steps", 1)
    )

    if args.global_batch_size is not None and args.grad_accum_steps is None:
        denom = micro_batch * visible_gpus
        if denom <= 0 or args.global_batch_size % denom != 0:
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size_per_gpu * visible_gpus"
            )
        grad_accum = args.global_batch_size // denom

    global_batch = micro_batch * grad_accum * visible_gpus

    if args.enforce_global_batch_size is not None and global_batch != args.enforce_global_batch_size:
        raise ValueError(
            f"Computed global_batch_size={global_batch} does not match enforce_global_batch_size="
            f"{args.enforce_global_batch_size}"
        )

    return micro_batch, grad_accum, global_batch


def graceful_distributed_shutdown():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            time.sleep(0.2)
        return True
    except Exception as e:
        print(
            f"[Rank {os.environ.get('RANK', '0')}] Warning: failed to destroy process group: {e}",
            flush=True,
        )
        return False


def build_launcher_metadata(trainer, checkpoint_path):
    import socket

    def collect_prefixed_env(prefixes):
        out = {}
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                out[key] = value
        return out

    slurm = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "nodelist": os.getenv("SLURM_NODELIST") or os.getenv("SLURM_JOB_NODELIST"),
        "hosts": os.getenv("SLURM_HOSTS"),
    }
    slurm = {k: v for k, v in slurm.items() if v}

    env_summary = {
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        "MASTER_ADDR": os.getenv("MASTER_ADDR"),
        "MASTER_PORT": os.getenv("MASTER_PORT"),
    }
    env_summary = {k: v for k, v in env_summary.items() if v}

    return {
        "checkpoint_path": checkpoint_path,
        "rank0_host": socket.gethostname(),
        "world_size": trainer.world_size,
        "git_commit": os.getenv("GIT_COMMIT") or trainer.metrics.get("git_commit", "unknown"),
        "slurm": slurm,
        "env": env_summary,
        "nccl_env": collect_prefixed_env(["NCCL_", "TORCH_NCCL_"]),
    }


def clean_distributed_shutdown(global_rank):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    if dist.is_available() and dist.is_initialized():
        try:
            timeout_sec = int(os.getenv("DIST_SHUTDOWN_TIMEOUT_SEC", "120"))
            backend = None
            try:
                backend = dist.get_backend()
            except Exception:
                backend = None

            if backend == "gloo" and hasattr(dist, "monitored_barrier"):
                from datetime import timedelta

                dist.monitored_barrier(timeout=timedelta(seconds=timeout_sec))
            else:
                # Best-effort time-bounded barrier to avoid end-of-run hangs.
                work = dist.barrier(async_op=True)
                if hasattr(work, "is_completed"):
                    deadline = time.time() + timeout_sec
                    while time.time() < deadline and not work.is_completed():
                        time.sleep(0.1)
                    if not work.is_completed():
                        print(
                            f"[Rank {global_rank}] Warning: shutdown barrier timed out after {timeout_sec}s; "
                            "continuing with destroy_process_group.",
                            flush=True,
                        )
                else:
                    # Fall back to a regular barrier if async completion can't be polled.
                    dist.barrier()
        except Exception as e:
            print(f"[Rank {global_rank}] Warning: dist.barrier failed during shutdown: {e}", flush=True)
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(
                f"[Rank {global_rank}] Warning: dist.destroy_process_group failed during shutdown: {e}",
                flush=True,
            )

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

        self.local_rank, self.global_rank, self.world_size = get_dist_info(args)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.current_device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.current_device = torch.device("cpu")

        distributed_env = (
            int(os.getenv("WORLD_SIZE", "1")) > 1
            or os.getenv("RANK") is not None
            or os.getenv("LOCAL_RANK") is not None
        )
        self.use_deepspeed = args.run_type == "optimized" or distributed_env
        if args.num_workers is not None:
            self.num_workers = int(args.num_workers)
        else:
            self.num_workers = 0 if distributed_env else 2
        if self.use_deepspeed and not dist.is_initialized():
            try:
                if self.args.quiet_nccl_monitor:
                    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
                    if self.global_rank == 0:
                        print("[Rank 0] NCCL monitoring disabled via TORCH_NCCL_ENABLE_MONITORING=0")
                import deepspeed
                print(f"[Rank {self.global_rank}] Initializing distributed process group via DeepSpeed...")
                deepspeed.init_distributed(dist_backend="nccl", init_method="env://")
                print(f"[Rank {self.global_rank}] Distributed process group initialized.")
            except Exception as e:
                print(f"[Rank {self.global_rank}] ERROR: DeepSpeed distributed init failed: {e}")
                sys.exit(1)

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if torch.cuda.is_available():
            current_cuda = torch.cuda.current_device()
        else:
            current_cuda = None
        print(
            f"[Rank {self.global_rank}] Startup: pid={os.getpid()} "
            f"rank={self.global_rank} local_rank={self.local_rank} world_size={self.world_size} "
            f"CUDA_VISIBLE_DEVICES={cuda_visible} torch.cuda.current_device={current_cuda}"
        )

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

        launcher = "deepspeed" if self.use_deepspeed else "baseline"
        self.rank_print(f"DeepSpeed enabled: {self.use_deepspeed}, launcher: {launcher}")

        try:
            with open(args.deepspeed_config, 'r') as f:
                deepspeed_config = json.load(f)
            self.rank_print(f"Loaded DeepSpeed config from {args.deepspeed_config}")
        except Exception as e:
             self.rank_print(f"ERROR: Failed to load DeepSpeed config: {e}")
             sys.exit(1)

        if args.micro_batch_size_per_gpu is not None:
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size_per_gpu
        if args.grad_accum_steps is not None:
            deepspeed_config["gradient_accumulation_steps"] = args.grad_accum_steps

        self.deepspeed_config = deepspeed_config
        visible_gpus = self.world_size if self.use_deepspeed else 1
        try:
            self.micro_batch_size, self.grad_accum_steps, self.global_batch_size = resolve_batch_config(
                args=self.args,
                ds_config=self.deepspeed_config,
                visible_gpus=visible_gpus,
            )
        except ValueError as e:
            self.rank_print(f"ERROR: {e}")
            sys.exit(1)

        self.deepspeed_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size
        self.deepspeed_config["gradient_accumulation_steps"] = self.grad_accum_steps

        if self.global_rank == 0:
            self.rank_print(
                "Run config: micro_batch_size_per_gpu=%s, grad_accum_steps=%s, visible_gpus=%s, "
                "global_batch_size=%s"
                % (self.micro_batch_size, self.grad_accum_steps, visible_gpus, self.global_batch_size)
            )

        self.use_amp = (
            bool(self.deepspeed_config.get("fp16", {}).get("enabled"))
            and torch.cuda.is_available()
            and not self.use_deepspeed
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        if self.deepspeed_config.get("fp16", {}).get("enabled"):
            self.precision_mode = "fp16_deepspeed" if self.use_deepspeed else "fp16_torch_amp"
        else:
            self.precision_mode = "fp32"

        if self.use_deepspeed:
            self.rank_print("Initializing DeepSpeed engine...")
            try:
                 import deepspeed
                 train_sampler = None
                 if self.world_size > 1:
                     train_sampler = DistributedSampler(
                         self.train_dataset,
                         num_replicas=self.world_size,
                         rank=self.global_rank,
                         shuffle=True,
                     )
                 train_dataloader = DataLoader(
                     self.train_dataset,
                     batch_size=self.micro_batch_size,
                     shuffle=train_sampler is None,
                     sampler=train_sampler,
                     num_workers=self.num_workers,
                     pin_memory=True,
                     drop_last=True,
                 )
                 try:
                     self.model_engine, self.optimizer, self.train_dataloader, self.scheduler = deepspeed.initialize(
                         model=self.model,
                         model_parameters=self.model.parameters(),
                         training_dataloader=train_dataloader,
                         config=deepspeed_config
                     )
                 except TypeError:
                     if self.global_rank == 0:
                         print(
                             "[Rank 0] DeepSpeed initialize() does not support training_dataloader; "
                             "falling back to training_data.",
                             flush=True,
                         )
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
        else:
            self.model_engine = self.model.to(self.current_device)
            optimizer_cfg = self.deepspeed_config.get("optimizer", {})
            optimizer_params = optimizer_cfg.get("params", {})
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_params.get("lr", 1e-4),
                betas=tuple(optimizer_params.get("betas", (0.9, 0.999))),
                eps=optimizer_params.get("eps", 1e-8),
                weight_decay=optimizer_params.get("weight_decay", 0.01),
            )
            self.scheduler = None
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.micro_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.rank_print("Initialized baseline optimizer and dataloader.")

        if dist.is_available() and dist.is_initialized():
             self.rank_print("Waiting at barrier after DeepSpeed initialization...")
             dist.barrier()
             self.rank_print("Passed barrier after DeepSpeed initialization.")

        if len(self.val_dataset) > 0:
            val_sampler = None
            if self.world_size > 1:
                val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=False,
                )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.micro_batch_size,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
            self.rank_print(f"Validation DataLoader created. Size: {len(self.val_dataloader)}")
        else:
             self.rank_print("Validation dataset is empty, skipping validation dataloader creation.")
             self.val_dataloader = None

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.metrics = metrics_utils.build_initial_metrics(
            args=self.args,
            ds_config=self.deepspeed_config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            world_size=self.world_size,
            repo_root=repo_root,
            ds_config_path=args.deepspeed_config,
            precision_override=self.precision_mode,
            micro_batch_size_per_gpu=self.micro_batch_size,
            grad_accum_steps=self.grad_accum_steps,
            global_batch_size=self.global_batch_size,
            quiet_nccl_monitor=self.args.quiet_nccl_monitor,
            deepspeed_enabled=self.use_deepspeed,
        )

        if args.resume:
             self.rank_print(f"Attempting to resume from checkpoint path: {args.checkpoint_path}")
             pass

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
            if dist.is_available() and dist.is_initialized():
                 dist.barrier()

            if torch.cuda.is_available():
                 torch.cuda.reset_peak_memory_stats(self.current_device)

            if self.global_rank == 0:
                 self.epoch_start_time = time.time()

            self.model_engine.train()
            if not self.use_deepspeed:
                self.optimizer.zero_grad(set_to_none=True)

            if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                 self.train_dataloader.sampler.set_epoch(epoch)
                 self.rank_print(f"Set dataloader sampler epoch to {epoch}")


            progress_bar = None
            if self.global_rank == 0:
                 progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.args.epochs} Rank 0")


            total_train_loss_epoch = 0.0
            num_tokens_processed_epoch = 0
            num_steps_epoch = 0
            step_time_samples = []
            dataload_time_samples = []
            cuda_step_time_samples = []
            timing_warmup_steps = 5
            timing_sample_every = 50

            if dist.is_available() and dist.is_initialized():
                 self.rank_print(f"Waiting at barrier: Start of Epoch {epoch+1} loop...")
                 dist.barrier()
                 self.rank_print(f"Passed barrier: Start of Epoch {epoch+1} loop.")


            train_iter = iter(self.train_dataloader)
            total_steps_in_epoch = len(self.train_dataloader)
            for step in range(total_steps_in_epoch):
                 if dist.is_available() and dist.is_initialized():
                      if step % 50 == 0:
                           print(f"[Rank {self.global_rank}] Waiting at barrier: Start of Step {step}, Epoch {epoch+1}")
                      dist.barrier()
                      if step % 50 == 0:
                           print(f"[Rank {self.global_rank}] Passed barrier: Start of Step {step}, Epoch {epoch+1}")


                 dataload_start = time.perf_counter()
                 try:
                      batch = next(train_iter)
                 except StopIteration:
                      break
                 dataload_end = time.perf_counter()
                 input_ids, targets = batch
                 input_ids = input_ids.to(self.current_device)
                 targets = targets.to(self.current_device)
                 inputs = (input_ids, None)
                 sample_timing = self.global_rank == 0 and step >= timing_warmup_steps and step % timing_sample_every == 0
                 start_event = None
                 end_event = None
                 if sample_timing and torch.cuda.is_available():
                      start_event = torch.cuda.Event(enable_timing=True)
                      end_event = torch.cuda.Event(enable_timing=True)

                 try:
                      step_start = time.perf_counter()
                      if start_event is not None:
                           start_event.record()
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Starting forward pass, Step {step}, Epoch {epoch+1}, Input Shape: {input_ids.shape}")
                      if self.use_deepspeed:
                           outputs = self.model_engine(*inputs)
                      else:
                           with torch.cuda.amp.autocast(enabled=self.use_amp):
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
                      batch_tokens = input_ids.numel()
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
                      if self.use_deepspeed:
                           self.model_engine.backward(loss)
                      else:
                           loss_to_backprop = loss / max(1, self.grad_accum_steps)
                           if self.use_amp:
                                self.scaler.scale(loss_to_backprop).backward()
                           else:
                                loss_to_backprop.backward()
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Completed backward pass, Step {step}, Epoch {epoch+1}")

                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Starting optimizer step, Step {step}, Epoch {epoch+1}")
                      if self.use_deepspeed:
                           self.model_engine.step()
                      else:
                           should_step = ((step + 1) % max(1, self.grad_accum_steps) == 0) or (step + 1 == total_steps_in_epoch)
                           if should_step:
                                grad_clip = self.deepspeed_config.get("gradient_clipping")
                                if grad_clip:
                                     if self.use_amp:
                                          self.scaler.unscale_(self.optimizer)
                                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                                if self.use_amp:
                                     self.scaler.step(self.optimizer)
                                     self.scaler.update()
                                else:
                                     self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                      if step % 50 == 0 or step < 5:
                           print(f"[Rank {self.global_rank}]: Completed optimizer step, Step {step}, Epoch {epoch+1}")
                      if end_event is not None:
                           end_event.record()
                      step_end = time.perf_counter()

                 except Exception as e:
                      print(f"[Rank {self.global_rank}]: CRITICAL ERROR during backward/step: Step {step}, Epoch {epoch+1}: {e}")
                      import traceback
                      traceback.print_exc()
                      if dist.is_available() and dist.is_initialized(): dist.barrier()
                      sys.exit(1)

                 if sample_timing:
                      step_time_samples.append(step_end - step_start)
                      dataload_time_samples.append(dataload_end - dataload_start)
                      if start_event is not None and end_event is not None:
                           torch.cuda.synchronize()
                           cuda_ms = start_event.elapsed_time(end_event)
                           cuda_step_time_samples.append(cuda_ms / 1000.0)

                 total_train_loss_epoch += batch_loss * batch_tokens
                 num_tokens_processed_epoch += batch_tokens
                 num_steps_epoch += 1

                 if self.global_rank == 0:
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

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if dist.is_available() and dist.is_initialized():
                loss_tokens = torch.tensor(
                    [total_train_loss_epoch, num_tokens_processed_epoch],
                    device=self.current_device,
                    dtype=torch.float64,
                )
                dist.all_reduce(loss_tokens, op=dist.ReduceOp.SUM)
                steps_tensor = torch.tensor(
                    [num_steps_epoch],
                    device=self.current_device,
                    dtype=torch.float64,
                )
                dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX)
                global_steps = int(steps_tensor.item())
                global_loss_sum = loss_tokens[0].item()
                global_tokens = loss_tokens[1].item()
            else:
                global_steps = num_steps_epoch
                global_loss_sum = total_train_loss_epoch
                global_tokens = num_tokens_processed_epoch

            max_alloc = 0
            max_reserved = 0
            max_alloc_global = None
            max_reserved_global = None
            if torch.cuda.is_available():
                max_alloc = torch.cuda.max_memory_allocated(self.current_device)
                max_reserved = torch.cuda.max_memory_reserved(self.current_device)
                if dist.is_available() and dist.is_initialized():
                    mem_tensor = torch.tensor(
                        [max_alloc, max_reserved],
                        device=self.current_device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
                    max_alloc_global = int(mem_tensor[0].item())
                    max_reserved_global = int(mem_tensor[1].item())

            if self.global_rank == 0:
                if progress_bar:
                    progress_bar.close()
                self.epoch_end_time = time.time()
                self.epoch_time = self.epoch_end_time - self.epoch_start_time

                avg_loss_epoch = global_loss_sum / global_tokens if global_tokens > 0 else 0.0
                train_throughput = global_tokens / self.epoch_time if self.epoch_time > 0 else 0.0

                self.rank_print(
                    f"Epoch {epoch+1} finished. Avg Loss: {avg_loss_epoch:.4f}, "
                    f"Time: {self.epoch_time:.2f}s, Throughput: {train_throughput:.2f} tokens/sec"
                )

                step_samples = cuda_step_time_samples if cuda_step_time_samples else step_time_samples
                step_time_mean = float(np.mean(step_samples)) if step_samples else None
                step_time_p50 = float(np.percentile(step_samples, 50)) if step_samples else None
                step_time_p95 = float(np.percentile(step_samples, 95)) if step_samples else None
                dataload_time_mean = float(np.mean(dataload_time_samples)) if dataload_time_samples else None

                grad_accum = self.deepspeed_config.get("gradient_accumulation_steps", 1)
                optimizer_steps = (global_steps + grad_accum - 1) // grad_accum if grad_accum else global_steps

                epoch_metrics = {
                    "epoch_idx": epoch + 1,
                    "epoch_wall_time_sec": self.epoch_time,
                    "steps": global_steps,
                    "micro_steps": global_steps,
                    "optimizer_steps": int(optimizer_steps),
                    "tokens_processed_global": int(global_tokens),
                    "tokens_per_sec_global": train_throughput,
                    "train_loss_avg_global": avg_loss_epoch,
                    "val_loss_avg_global": None,
                    "max_cuda_mem_allocated_bytes_rank0": int(max_alloc),
                    "max_cuda_mem_reserved_bytes_rank0": int(max_reserved),
                    "step_time_mean_sec": step_time_mean,
                    "step_time_p50_sec": step_time_p50,
                    "step_time_p95_sec": step_time_p95,
                    "dataload_time_mean_sec": dataload_time_mean,
                }
                if max_alloc_global is not None:
                    epoch_metrics["max_cuda_mem_allocated_bytes_global"] = max_alloc_global
                    epoch_metrics["max_cuda_mem_reserved_bytes_global"] = max_reserved_global

                self.metrics["epochs"].append(epoch_metrics)

            if dist.is_available() and dist.is_initialized():
                self.rank_print(f"Waiting at barrier before validation, Epoch {epoch+1}...")
                dist.barrier()
                self.rank_print(f"Passed barrier, starting validation, Epoch {epoch+1}.")

            if self.val_dataloader:
                val_loss_avg = self.validate(epoch)
                if self.global_rank == 0 and self.metrics["epochs"]:
                    self.metrics["epochs"][-1]["val_loss_avg_global"] = val_loss_avg
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
                     if self.use_deepspeed:
                          self.model_engine.save_16bit_model(self.args.checkpoint_path, tag)
                          self.rank_print(f"Checkpoint saved successfully with tag '{tag}'")
                     else:
                          os.makedirs(self.args.checkpoint_path, exist_ok=True)
                          ckpt_path = os.path.join(self.args.checkpoint_path, f"{tag}.pt")
                          torch.save(self.model.state_dict(), ckpt_path)
                          self.rank_print(f"Checkpoint saved successfully to {ckpt_path}")
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

            per_epoch = self.metrics.get("epochs", [])
            if per_epoch:
                mean_tokens_per_sec = sum(
                    epoch.get("tokens_per_sec_global", 0.0) for epoch in per_epoch
                ) / len(per_epoch)
                val_losses = [
                    epoch.get("val_loss_avg_global")
                    for epoch in per_epoch
                    if epoch.get("val_loss_avg_global") is not None
                ]
                best_val_loss = min(val_losses) if val_losses else None
            else:
                mean_tokens_per_sec = None
                best_val_loss = None

            self.metrics["summary"]["total_wall_time_sec"] = self.total_training_time
            self.metrics["summary"]["best_val_loss"] = best_val_loss
            self.metrics["summary"]["mean_tokens_per_sec_global"] = mean_tokens_per_sec


    def validate(self, epoch):
        self.rank_print(f"Starting validation for Epoch {epoch+1}")
        self.model_engine.eval()

        total_loss_sum = 0.0
        total_tokens = 0

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
                      if self.use_deepspeed:
                           outputs = self.model_engine(*inputs)
                      else:
                           with torch.cuda.amp.autocast(enabled=self.use_amp):
                                outputs = self.model_engine(*inputs)
                      if step % 20 == 0 or step < 3:
                          print(f"[Rank {self.global_rank}]: Validation Step {step}, Epoch {epoch+1}, Output Shape: {outputs.shape}")

                      loss = F.cross_entropy(outputs.view(-1, self.args.vocab_size), targets.view(-1))
                      batch_loss = loss.item()
                      batch_tokens = input_ids.numel()
                      if step % 20 == 0 or step < 3:
                           print(f"[Rank {self.global_rank}]: Validation Loss: {batch_loss:.4f}, Step {step}, Epoch {epoch+1}")

                      total_loss_sum += batch_loss * batch_tokens
                      total_tokens += batch_tokens
                      if self.global_rank == 0 and val_progress_bar:
                          val_progress_bar.update(1)
                          val_progress_bar.set_postfix(loss=batch_loss)

                 except Exception as e:
                      print(f"[Rank {self.global_rank}]: CRITICAL ERROR during validation: Step {step}, Epoch {epoch+1}: {e}")
                      import traceback
                      traceback.print_exc()
                      if dist.is_available() and dist.is_initialized(): dist.barrier()
                      break

        if dist.is_available() and dist.is_initialized():
            loss_tokens = torch.tensor(
                [total_loss_sum, total_tokens],
                device=self.current_device,
                dtype=torch.float64,
            )
            dist.all_reduce(loss_tokens, op=dist.ReduceOp.SUM)
            global_loss_sum = loss_tokens[0].item()
            global_tokens = loss_tokens[1].item()
        else:
            global_loss_sum = total_loss_sum
            global_tokens = total_tokens

        avg_loss = global_loss_sum / global_tokens if global_tokens > 0 else 0.0

        if self.global_rank == 0:
            if val_progress_bar:
                val_progress_bar.close()
            self.rank_print(f"Validation Finished Epoch {epoch+1}. Average Loss: {avg_loss:.4f}")
            return avg_loss

        if dist.is_available() and dist.is_initialized():
             self.rank_print(f"Waiting at barrier after validation, Epoch {epoch+1}...")
             dist.barrier()
             self.rank_print(f"Passed barrier after validation, Epoch {epoch+1}.")
        return None


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

        return generated_text, generate_time, throughput

    def save_metrics(self, metrics, file_path):
        if self.global_rank == 0:
            self.rank_print(f"Saving metrics to {file_path}")
            try:
                metrics_utils.write_json_atomic(file_path, metrics)
            except Exception as e:
                 self.rank_print(f"ERROR: Failed to save metrics to {file_path}: {e}")


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
    parser.add_argument('--micro_batch_size_per_gpu', type=int, default=None, help='Override micro batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=None, help='Override gradient accumulation steps')
    parser.add_argument('--global_batch_size', type=int, default=None, help='Override global batch size')
    parser.add_argument('--enforce_global_batch_size', type=int, default=None, help='Fail if computed global batch size differs')
    parser.add_argument('--quiet_nccl_monitor', action='store_true', help='Disable NCCL monitoring in torch.distributed')
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
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int, default=-1, help='Local rank passed by launcher')

    parser.add_argument('--run_type', type=str, choices=['baseline', 'optimized'], required=True, help='Type of run')

    # parser.add_argument('--pipeline_stages', type=int, default=1, help='Number of pipeline stages (overridden by deepspeed config)')

    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler (logs saved in checkpoint_path/profiler_logs)')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader worker processes (default: 0 for distributed, 2 otherwise)')


    args = parser.parse_args()

    if args.quiet_nccl_monitor:
         os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"

    env_rank = os.environ.get("RANK")
    rank = int(env_rank) if env_rank is not None else 0
    env_world_size = os.environ.get("WORLD_SIZE")
    world_size = int(env_world_size) if env_world_size is not None else 1
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank_arg = getattr(args, "local_rank", None)
    local_rank_missing = local_rank_env is None and (local_rank_arg is None or int(local_rank_arg) < 0)
    if rank == 0 and world_size <= 1 and local_rank_missing:
         print("Warning: local_rank not set by launcher, running in non-distributed mode.")

    trainer = None
    global_rank_for_shutdown = rank
    try:
        trainer = GPT2Trainer(args)
        global_rank_for_shutdown = trainer.global_rank

        trainer.train()

        if trainer.global_rank == 0:
            launcher_metadata_path = os.path.join(args.checkpoint_path, "launcher_metadata.json")
            metrics_utils.write_json_atomic(
                launcher_metadata_path,
                build_launcher_metadata(trainer=trainer, checkpoint_path=args.checkpoint_path),
            )
            print(f"[Rank 0] LAUNCHER_METADATA_WRITE_DONE path={launcher_metadata_path}", flush=True)

        if trainer.global_rank == 0:
            timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            trainer.metrics["completion"] = {
                "run_complete_file": "RUN_COMPLETE.txt",
                "printed_marker": True,
                "timestamp_utc": timestamp_utc,
            }
            trainer.metrics["shutdown"]["destroy_process_group_called"] = dist.is_available() and dist.is_initialized()
            trainer.metrics["shutdown"]["quiet_nccl_monitor"] = args.quiet_nccl_monitor

            metrics_file_path = os.path.join(args.checkpoint_path, "training_metrics.json")
            trainer.save_metrics(trainer.metrics, metrics_file_path)
            print(f"[Rank 0] METRICS_WRITE_DONE path={metrics_file_path}", flush=True)

        if trainer.global_rank == 0:
            tokens_per_sec = trainer.metrics.get("summary", {}).get("mean_tokens_per_sec_global")
            total_wall_time = trainer.metrics.get("summary", {}).get("total_wall_time_sec")
            run_complete_line = (
                f"[Rank 0] RUN_COMPLETE checkpoint_path={args.checkpoint_path} "
                f"world_size={trainer.world_size} tokens_per_sec={tokens_per_sec} "
                f"total_wall_time_sec={total_wall_time}"
            )
            run_complete_path = os.path.join(args.checkpoint_path, "RUN_COMPLETE.txt")
            os.makedirs(args.checkpoint_path, exist_ok=True)
            with open(run_complete_path, "w", encoding="utf-8") as f:
                f.write(f"{run_complete_line} timestamp_utc={timestamp_utc}\n")
            print(run_complete_line, flush=True)
    finally:
        clean_distributed_shutdown(global_rank=global_rank_for_shutdown)
        print(f"[Rank {global_rank_for_shutdown}] EXITING cleanly", flush=True)

if __name__ == '__main__':
    main()
