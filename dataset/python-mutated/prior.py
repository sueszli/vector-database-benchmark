import math
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['Prior']

def sinusoidal_embedding(timesteps, dim):
    if False:
        for i in range(10):
            print('nop')
    half = dim // 2
    timesteps = timesteps.float()
    sinusoid = torch.outer(timesteps, torch.pow(10000, -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x

class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        if False:
            print('Hello World!')
        assert dim % num_heads == 0
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.pow(self.head_dim, -0.25)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        if False:
            print('Hello World!')
        (b, l, n, c) = (*x.shape[:2], self.num_heads, self.head_dim)
        (q, k, v) = self.to_qkv(x).view(b, l, n * 3, c).chunk(3, dim=2)
        attn = torch.einsum('binc,bjnc->bnij', q * self.scale, k * self.scale)
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :l, :l] == 0, float('-inf'))
        attn = F.softmax(attn.float(), dim=-1).type(attn.dtype)
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        x = x.reshape(b, l, -1)
        x = self.proj(x)
        return x

class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads):
        if False:
            return 10
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, mask=None):
        if False:
            i = 10
            return i + 15
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class Prior(nn.Module):

    def __init__(self, dim=2048, clip_dim=768, num_heads=32, num_layers=24):
        if False:
            return 10
        super(Prior, self).__init__()
        self.dim = dim
        self.clip_dim = clip_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_embedding = nn.Sequential(nn.Linear(clip_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.vision_embedding = nn.Sequential(nn.Linear(clip_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.eos_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 4, dim))
        self.blocks = nn.ModuleList([AttentionBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, clip_dim)
        self.register_buffer('attn_mask', torch.tril(torch.ones(1, 1, 4, 4)))
        self.init_weights()

    def forward(self, x, t, y):
        if False:
            print('Hello World!')
        'x:      [B, C].\n            t:      [B].\n            y:      [B, C].\n        '
        b = x.size(0)
        u1 = sinusoidal_embedding(t, self.dim)
        u2 = [self.text_embedding(y).unsqueeze(1), self.time_embedding(u1).unsqueeze(1), self.vision_embedding(x).unsqueeze(1), self.eos_embedding.repeat(b, 1, 1)]
        x = self.pos_embedding + torch.cat(u2, dim=1)
        for block in self.blocks:
            x = block(x, self.attn_mask)
        x = self.norm(x)
        x = self.head(x[:, -1])
        return x

    def init_weights(self):
        if False:
            return 10
        std = 0.02 / math.sqrt(2.0 * self.num_layers)
        for (name, m) in self.named_modules():
            if name.endswith('attn.proj') or name.endswith('ffn.2'):
                nn.init.normal_(m.weight, std=std)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def param_groups(self):
        if False:
            print('Hello World!')
        groups = [{'params': [p for (n, p) in self.named_parameters() if 'norm' in n or n.endswith('bias')], 'weight_decay': 0.0}, {'params': [p for (n, p) in self.named_parameters() if not ('norm' in n or n.endswith('bias'))]}]
        return groups