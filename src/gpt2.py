import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import multiprocessing
from functools import partial

from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import concurrent.futures

# Define a global tokenizer name to be used in subprocesses
TOKENIZER_NAME = 'gpt2'

def process_single_file(file_path, tokenizer_name, seq_length, max_tokens_per_file):
    """
    Reads a single text file, tokenizes its content, truncates if necessary,
    splits into fixed-length sequences, and returns the list of sequences.

    Args:
        file_path (str): Path to the text file.
        tokenizer_name (str): Name of the pretrained tokenizer.
        seq_length (int): Length of each token sequence.
        max_tokens_per_file (int): Maximum number of tokens per file.

    Returns:
        List[List[int]]: A list of token sequences.
    """
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens_len = len(tokens)
            if tokens_len == 0:
                return []
            if tokens_len > max_tokens_per_file:
                print(f"File {file_path} has {tokens_len} tokens, exceeding the max {max_tokens_per_file}. Truncating.")
                tokens = tokens[:max_tokens_per_file]
            # Split tokens into sequences of seq_length
            num_sequences = (len(tokens) - seq_length) // seq_length
            sequences = []
            for i in range(num_sequences):
                sequence = tokens[i * seq_length : (i + 1) * seq_length]
                sequences.append(sequence)
            return sequences
    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        return []

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
        Q = self.query(x)  # (batch_size, seq_length, embedding_size)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_length, seq_length)
        
        # Apply attention mask (e.g., causal mask for autoregressive modeling)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Attention output
        attn_output = torch.matmul(attn_probs, V)  # (batch_size, num_heads, seq_length, head_dim)
        
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
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embedding_size, eps=1e-5)
        self.head = nn.Linear(embedding_size, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        # Get batch size and sequence length
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)
        
        # Attention mask (for causal masking)
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        
        # Assert sequence length does not exceed max_position_embeddings
        assert seq_length <= self.position_embedding.position_embeddings.num_embeddings, \
            f"Sequence length {seq_length} exceeds max_position_embeddings {self.position_embedding.position_embeddings.num_embeddings}"
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.head(hidden_states)
        
        return logits

# Training and Checkpointing

class GPT2Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model hyperparameters
        self.vocab_size = 50257     # GPT-2 vocabulary size
        self.embedding_size = 768   # Embedding size for tokens and positions
        self.num_layers = 12        # Number of transformer blocks
        self.num_heads = 12         # Number of attention heads
        self.dropout = 0.1          # Dropout rate
        self.max_position_embeddings = 1024  # Maximum sequence length
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.model_max_length = self.max_position_embeddings  # Ensure tokenizer does not exceed max_position_embeddings
        
        # Initialize model
        self.model = GPT2Model(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_position_embeddings=self.max_position_embeddings
        ).to(self.device)
        
        # Define loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        
        # Initialize dataset and dataloader
        self.seq_length = 32  # Adjust as needed
        self.dataset = self.initialize_dataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        # Initialize starting epoch
        self.start_epoch = 0
        
        # Load checkpoint if resume is set
        if args.resume and os.path.exists(args.checkpoint_path):
            self.load_checkpoint(args.checkpoint_path)
        
        # Check if dataset has any examples
        if len(self.dataset) == 0:
            raise ValueError("The dataset is empty. Please ensure that the preprocessing step was successful and that there are .txt files in the specified text_dir.")
    
    def initialize_dataset(self):
        return TextDataset(
            text_dir=self.args.text_dir,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            max_tokens_per_file=100000
        )
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {self.start_epoch}")
    
    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for batch in progress_bar:
                self.optimizer.zero_grad()
                input_ids = batch.to(self.device)
                
                # Assert sequence length does not exceed max_position_embeddings
                assert input_ids.size(1) <= self.max_position_embeddings, \
                    f"Sequence length {input_ids.size(1)} exceeds max_position_embeddings {self.max_position_embeddings}"
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Compute loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = self.loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(self.dataloader)
            print(f'Epoch {epoch+1}/{self.args.epochs}, Average Loss: {avg_loss:.4f}')
    
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(checkpoint, self.args.checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    def generate_text(self, prompt, max_length=50):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Ensure input does not exceed max_position_embeddings
                if input_ids.size(1) > self.max_position_embeddings - 1:
                    input_ids = input_ids[:, - (self.max_position_embeddings - 1):]
                
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                
                # Stop if EOS token is generated
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
        return generated_text

class TextDataset(Dataset):
    def __init__(self, text_dir, tokenizer, seq_length, max_tokens_per_file=100000):
        """
        Initializes the dataset by listing all text files in text_dir and processing them into fixed-length sequences
        using multiprocessing for faster tokenization and loading.

        Args:
            text_dir (str): Directory containing text files.
            tokenizer (GPT2Tokenizer): Tokenizer to encode text.
            seq_length (int): Length of each token sequence.
            max_tokens_per_file (int): Maximum number of tokens per file.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.file_paths = []
        for root, dirs, files in os.walk(text_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.text'):
                    self.file_paths.append(os.path.join(root, file))
        
        print(f"Total text files found: {len(self.file_paths)}")
        
        self.examples = []
        
        # Use multiprocessing to process files in parallel
        num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers or number of CPU cores
        print(f"Using {num_workers} worker(s) for tokenization and loading.")
        
        # Define a partial function with fixed arguments
        process_func = partial(
            process_single_file,
            tokenizer_name=TOKENIZER_NAME,
            seq_length=self.seq_length,
            max_tokens_per_file=max_tokens_per_file
        )
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all file processing tasks
            futures = {executor.submit(process_func, file_path): file_path for file_path in self.file_paths}
            
            # Use tqdm to display progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Tokenizing text files"):
                file_path = futures[future]
                try:
                    sequences = future.result()
                    self.examples.extend(sequences)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        
        print(f"Total training examples: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

# Example text generation utility
def generate_text(model, tokenizer, prompt, max_length=50, device='cpu'):
    """
    Generates text using the trained GPT-2 model.

    Args:
        model (GPT2Model): Trained GPT-2 model.
        tokenizer (GPT2Tokenizer): Tokenizer to encode and decode text.
        prompt (str): Initial text prompt.
        max_length (int): Maximum number of tokens to generate.
        device (str): Device to run the model on.

    Returns:
        str: Generated text.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Ensure input does not exceed max_position_embeddings
            if input_ids.size(1) > model.max_position_embeddings - 1:
                input_ids = input_ids[:, - (model.max_position_embeddings - 1):]
            
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 Model with Multiprocessing')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, default='gpt2_checkpoint.pt', help='Path to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--text_dir', type=str, default='text_data/', help='Directory containing preprocessed text files')
    args = parser.parse_args()
    
    trainer = GPT2Trainer(args)
    trainer.train()
    
    # Generate text after training
    prompt = "Hello. How are you?"
    generated_text = generate_text(trainer.model, trainer.tokenizer, prompt, max_length=50, device=trainer.device)
    print("Generated Text:")
    print(generated_text)

if __name__ == '__main__':
    main()
