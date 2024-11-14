import torch
from torch import nn
from einops import einsum, rearrange
from lightning import LightningModule


class Evaluator(LightningModule):
    def __init__(self, dim, n_heads, n_layers, lr):
        super().__init__()
        self.embedding = torch.nn.Embedding(5 * 32, dim)
        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, seq_len=32) for _ in range(n_layers)
        ])
        self.out = nn.Linear(dim, 1)
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).mean(dim=1).squeeze()
        loss = self.loss(y_hat, y.squeeze())
        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, seq_len=None):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, seq_len=seq_len)
        self.pos_ff = PositionwiseFeedForward(dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.pos_ff(self.n2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, seq_len=None):
        super().__init__()
        self.n_heads = n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, mask=None):
        q = rearrange(self.wq(x), "b n (h d) -> b n h d", h=self.n_heads)
        k = rearrange(self.wk(x), "b n (h d) -> b n h d", h=self.n_heads)
        q = rearrange(q, "b n h d -> b h n d")
        k = rearrange(k, "b n h d -> b h n d")
        
        v = rearrange(self.wv(x), "b n (h d) -> b h n d", h=self.n_heads)
        
        similarity = einsum(q, k, "b h n d, b h s d -> b h n s")
        if mask is not None:
            attention =  nn.functional.softmax(similarity + mask.unsqueeze(1), dim=-1) / (k.size(-1) ** 0.5)
        else:
            attention = nn.functional.softmax(similarity, dim=-1) / (k.size(-1) ** 0.5)
        x = einsum(attention, v, "b h n s, b h s d -> b h n d")
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out_proj(self.norm(x))

        return x

    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.SiLU(),
            nn.LayerNorm(2*dim),
            nn.Linear(2*dim, dim),
        )
        
        self._init()
        
    def _init(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[3].weight)
        
    def forward(self, x):
        return self.net(x) + x

