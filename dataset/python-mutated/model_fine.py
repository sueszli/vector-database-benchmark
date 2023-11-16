"""
Much of this code is adapted from Andrej Karpathy's NanoGPT
(https://github.com/karpathy/nanoGPT)
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .model import GPT, MLP, GPTConfig

class NonCausalSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        (B, T, C) = x.size()
        (q, k, v) = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        else:
            att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class FineBlock(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class FineGPT(GPT):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        del self.lm_head
        self.config = config
        self.n_codes_total = config.n_codes_total
        self.transformer = nn.ModuleDict(dict(wtes=nn.ModuleList([nn.Embedding(config.input_vocab_size, config.n_embd) for _ in range(config.n_codes_total)]), wpe=nn.Embedding(config.block_size, config.n_embd), drop=nn.Dropout(config.dropout), h=nn.ModuleList([FineBlock(config) for _ in range(config.n_layer)]), ln_f=nn.LayerNorm(config.n_embd)))
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.output_vocab_size, bias=False) for _ in range(config.n_codes_given, self.n_codes_total)])
        for i in range(self.n_codes_total - config.n_codes_given):
            self.transformer.wtes[i + 1].weight = self.lm_heads[i].weight

    def forward(self, pred_idx, idx):
        if False:
            for i in range(10):
                print('nop')
        device = idx.device
        (b, t, codes) = idx.size()
        assert t <= self.config.block_size, f'Cannot forward sequence of length {t}, block size is only {self.config.block_size}'
        assert pred_idx > 0, 'cannot predict 0th codebook'
        assert codes == self.n_codes_total, (b, t, codes)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_embs = [wte(idx[:, :, i]).unsqueeze(-1) for (i, wte) in enumerate(self.transformer.wtes)]
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb[:, :, :, :pred_idx + 1].sum(dim=-1)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits

    def get_num_params(self, non_embedding=True):
        if False:
            i = 10
            return i + 15
        '\n        Return the number of parameters in the model.\n        For non-embedding count (default), the position embeddings get subtracted.\n        The token embeddings would too, except due to the parameter sharing these\n        params are actually used as weights in the final layer, so we include them.\n        '
        n_params = sum((p.numel() for p in self.parameters()))
        if non_embedding:
            for wte in self.transformer.wtes:
                n_params -= wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1