import torch

class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half):
        if False:
            i = 10
            return i + 15
        'Rotary positional embedding\n        Reference : https://blog.eleuther.ai/rotary-embeddings/\n        Paper: https://arxiv.org/pdf/2104.09864.pdf\n        Args:\n            dim: Dimension of embedding\n            base: Base value for exponential\n            precision: precision to use for numerical values\n        '
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.precision = precision

    def forward(self, x, seq_len: int=0):
        if False:
            print('Hello World!')
        '\n        Args:\n            x: Input x with T X B X C\n            seq_len: Sequence length of input x\n        '
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
            self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
        return (self.cos_cached, self.sin_cached)

def rotate_half(x):
    if False:
        print('Hello World!')
    (x1, x2) = (x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int=0):
    if False:
        print('Hello World!')
    (cos, sin) = (cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...])
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)