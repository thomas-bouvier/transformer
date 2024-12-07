import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
bs = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping from characters to integers
stoi = { s: i for i, s in enumerate(chars) }
itos = { i: s for i, s in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (bs, )) # get random offsets
    x = torch.stack([data[i:i + block_size] for i in ix]) # stack by row
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval() # does not change much as we don't have batch norm layers

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out


class MultiHead(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Self-attention head
        #self.sa_head = Head(n_embd)
        self.sa_heads = MultiHead(4, n_embd // 4) # 4 heads of 8-dimensional self-attention

        # As we are using embeddings and not bigrams anymore,
        # a linear layer is needed.
        self.lm_head = nn.Linear(n_embd, vocab_size) # (B, T, vocab_size)

        # Embeddings encode token identities
        # Positional embeddings encode their position
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # There are vocab_size channels
        # When using embeddings the following doesn't return logits anymore
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channel)

        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # Using the linear layer to get logits
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # (B, T, head_size)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # cross entropy expects the data to be reshaped
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # we want to measure the quality of these logits
            # the negative log likelihood does that
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices
        for _ in range(max_new_tokens):
            # look at the block_size context only
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Training loop
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_losses()
        print(f"Step {step}: train loss {losses['train']}, val loss {losses['val']}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))
