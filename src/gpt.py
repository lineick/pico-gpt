# train_model.py
import argparse
import os
import sys
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_training_data(train_dir):
    """Reads and concatenates all .txt files in the training directory."""
    text = ""
    for filename in sorted(os.listdir(train_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(train_dir, filename)
            print(f"Loading {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                text += f.read() + "\n"
    return text


def get_batch(data, block_size, batch_size, device):
    """Generates a batch of data for training."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        wei = (q @ k.transpose(-2, -1)) * (C**-0.5)  # Scaled dot-product attention
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # Mask future positions
        wei = F.softmax(wei, dim=-1)  # Softmax over last dimension
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple feed-forward network."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """The GPT Language Model."""

    def __init__(
        self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))  # (B*T,vocab_size)
            targets = targets.view(-1)  # (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generates new text based on the context idx."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]  # Crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B,vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B,vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


def train_model(args, max_iters=1000):
    """Trains the GPT language model based on provided arguments."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters (can be customized)
    batch_size = 64
    block_size = 256
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    dropout = 0.2
    torch.manual_seed(1337)

    # Load data
    text = load_training_data(args.train)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]  # Encoder function
    decode = lambda l: "".join([itos.get(i, "") for i in l])  # Decoder function

    # Save the vocab mappings for later use
    torch.save({"stoi": stoi, "itos": itos}, "vocab.pth")

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Initialize model with hyperparameters
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=block_size,
        dropout=dropout,
        device=device,
    ).to(device)
    print(
        f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Interrupt handling
    interrupted = False

    def handle_interrupt(signal_num, frame):
        nonlocal interrupted
        print("\nInterrupt received. Training will stop after the current iteration.")
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)

    # Estimate loss function
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            data_split = train_data if split == "train" else val_data
            for k in range(eval_iters):
                xb, yb = get_batch(data_split, block_size, batch_size, device)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training loop
    try:
        for iter in range(max_iters):
            if interrupted:
                break  # Exit loop if interrupted

            # Evaluate loss periodically
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(
                    f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

            # Get batch and perform forward and backward pass
            xb, yb = get_batch(train_data, block_size, batch_size, device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Save the model upon interruption or completion
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "stoi": stoi,
                "itos": itos,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "block_size": block_size,
                "dropout": dropout,
            },
            args.o,
        )
        print(f"Model saved to {args.o}")
        if interrupted:
            print("Training was interrupted and the model was saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-like language model.")
    parser.add_argument(
        "--train", type=str, required=True, help="Path to the training data directory."
    )
    parser.add_argument(
        "-o", type=str, required=True, help="Output path to save the trained model."
    )
    parser.add_argument(
        "--n_embd", type=int, default=384, help="Embedding size (default: 384)"
    )
    parser.add_argument(
        "--n_head", type=int, default=6, help="Number of heads (default: 6)"
    )
    parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers (default: 6)"
    )
    parser.add_argument("--iters", type=int, default=1000, help="training iters")
    args = parser.parse_args()

    # Enforce that the output path is a specific filename ending with .pth
    if not args.o.endswith(".pth"):
        print(
            "Error: Please specify a filename ending with '.pth' for the output path using '-o'."
        )
        sys.exit(1)

    train_model(args, args.iters)
