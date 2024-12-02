import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

# Import necessary modules for mixed-precision training
from torch.cuda.amp import GradScaler, autocast

# Import AdamW optimizer and learning rate scheduler
from transformers import AdamW, get_linear_schedule_with_warmup

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
    def __init__(self, embedding_size, num_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()
        
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask (e.g., causal mask for autoregressive modeling)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Attention output
        attn_output = torch.matmul(attn_probs, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_length, embed_dim)
        
        # Final linear layer
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
    def __init__(self, embedding_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_size, eps=1e-5)
        self.attn = MultiHeadSelfAttention(embedding_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embedding_size, eps=1e-5)
        self.ffn = FeedForward(embedding_size, dropout)
        
    def forward(self, x, attention_mask=None):
        # Multi-head self-attention with residual connection
        x = x + self.attn(self.ln1(x), attention_mask)
        # Feed-forward network with residual connection
        x = x + self.ffn(self.ln2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, dropout, max_position_embeddings):
        super(GPT2Model, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_size)
        self.position_embedding = PositionalEmbedding(max_position_embeddings, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.max_position_embeddings = max_position_embeddings
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, num_heads, dropout) for _ in range(num_layers)
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
        
        # Causal mask for autoregressive modeling
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
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
        self.device = device

        # Model hyperparameters
        self.vocab_size = 50257     # GPT-2 vocabulary size
        self.embedding_size = 768   # Embedding size for tokens and positions
        self.num_layers = 12        # Number of transformer blocks
        self.num_heads = 12         # Number of attention heads
        self.dropout = 0.1          # Dropout rate
        self.max_position_embeddings = 1024  # Maximum sequence length

        # Initialize tokenizer for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Set sequence length
        self.seq_length = 1024  # Must match the preprocessing script

        # Initialize model
        self.model = GPT2Model(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_position_embeddings=self.max_position_embeddings
        ).to(self.device)

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize optimizer with weight decay
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # Initialize dataset and dataloader
        train_data_path = args.train_data_path
        val_data_path = args.val_data_path

        self.train_dataset = BinaryDataset(train_data_path, self.seq_length)
        self.val_dataset = BinaryDataset(val_data_path, self.seq_length)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True
        )

        # Initialize learning rate scheduler
        total_steps = len(self.train_dataloader) * args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Initialize gradient scaler for mixed-precision training
        self.scaler = GradScaler()

        # Initialize starting epoch
        self.start_epoch = 0

        # Load checkpoint if resume is set
        if args.resume and os.path.exists(args.checkpoint_path):
            self.load_checkpoint(args.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for batch in progress_bar:
                input_ids, targets = batch  # Unpack the batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Mixed-precision training
                with autocast():
                    logits = self.model(input_ids)
                    loss = self.loss_fn(logits.view(-1, self.vocab_size), targets.view(-1))

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Scheduler step
                self.scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_dataloader)
            print(f'Epoch {epoch+1}/{self.args.epochs}, Average Training Loss: {avg_loss:.4f}')

            # Validate the model
            self.validate()

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }
            torch.save(checkpoint, self.args.checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, targets in self.val_dataloader:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                # Mixed-precision inference
                with autocast():
                    logits = self.model(input_ids)
                    loss = self.loss_fn(logits.view(-1, self.vocab_size), targets.view(-1))

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        print(f'Validation Loss: {avg_loss:.4f}')

    def generate_text(self, prompt, max_length=50):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        generated = input_ids
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = generated[:, -self.max_position_embeddings:]

                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token_id], dim=1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        generated_text = self.tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
        return generated_text

def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 Model with Binary Dataset')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--checkpoint_path', type=str, default='gpt2_checkpoint.pt', help='Path to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--train_data_path', type=str, default='train.bin', help='Path to the training binary file')
    parser.add_argument('--val_data_path', type=str, default='val.bin', help='Path to the validation binary file')
    args = parser.parse_args()

    trainer = GPT2Trainer(args)
    trainer.train()

    # Generate text after training
    prompt = "Once upon a time"
    generated_text = trainer.generate_text(prompt, max_length=50)
    print("Generated Text:")
    print(generated_text)

if __name__ == '__main__':
    main()
