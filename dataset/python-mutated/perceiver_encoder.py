from collections import namedtuple
from functools import wraps
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from packaging import version
from torch import einsum, nn

def exists(val):
    if False:
        for i in range(10):
            print('nop')
    return val is not None

def once(fn):
    if False:
        for i in range(10):
            print('nop')
    called = False

    @wraps(fn)
    def inner(x):
        if False:
            for i in range(10):
                print('nop')
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner
print_once = once(print)

class Attend(nn.Module):

    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.register_buffer('mask', None, persistent=False)
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not use_flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = self.config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n, device):
        if False:
            while True:
                i = 10
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer('mask', mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        if False:
            i = 10
            return i + 15
        (_, heads, q_len, _, k_len, is_cuda) = (*q.shape, k.shape[-2], q.is_cuda)
        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)
        config = self.cuda_config if is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        return out

    def forward(self, q, k, v, mask=None):
        if False:
            print('Hello World!')
        '\n        einstein notation\n        b - batch\n        h - heads\n        n, i, j - sequence length (base sequence length, source, target)\n        d - feature dimension\n        '
        (n, device) = (q.shape[-2], q.device)
        scale = q.shape[-1] ** (-0.5)
        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        return out

def Sequential(*mods):
    if False:
        return 10
    return nn.Sequential(*filter(exists, mods))

def exists(x):
    if False:
        print('Hello World!')
    return x is not None

def default(val, d):
    if False:
        for i in range(10):
            print('nop')
    if exists(val):
        return val
    return d() if callable(d) else d

class RMSNorm(nn.Module):

    def __init__(self, dim, scale=True, dim_cond=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        if False:
            print('Hello World!')
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma
        if not self.cond:
            return out
        assert exists(cond)
        (gamma, beta) = self.to_gamma_beta(cond).chunk(2, dim=-1)
        (gamma, beta) = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta

class CausalConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride
        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)

class GEGLU(nn.Module):

    def forward(self, x):
        if False:
            return 10
        (x, gate) = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

def FeedForward(dim, mult=4, causal_conv=False):
    if False:
        print('Hello World!')
    dim_inner = int(dim * mult * 2 / 3)
    conv = None
    if causal_conv:
        conv = nn.Sequential(Rearrange('b n d -> b d n'), CausalConv1d(dim_inner, dim_inner, 3), Rearrange('b d n -> b n d'))
    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), conv, nn.Linear(dim_inner, dim))

class PerceiverResampler(nn.Module):

    def __init__(self, *, dim, depth=2, dim_context=None, num_latents=32, dim_head=64, heads=8, ff_mult=4, use_flash_attn=False):
        if False:
            return 10
        super().__init__()
        dim_context = default(dim_context, dim)
        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash_attn, cross_attn_include_queries=True), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        if False:
            for i in range(10):
                print('nop')
        batch = x.shape[0]
        x = self.proj_context(x)
        latents = repeat(self.latents, 'n d -> b n d', b=batch)
        for (attn, ff) in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)

class Attention(nn.Module):

    def __init__(self, dim, *, dim_context=None, causal=False, dim_head=64, heads=8, dropout=0.0, use_flash=False, cross_attn_include_queries=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)
        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        if False:
            return 10
        (h, has_context) = (self.heads, exists(context))
        context = default(context, x)
        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)
        (q, k, v) = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        (q, k, v) = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        out = self.attend(q, k, v, mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)