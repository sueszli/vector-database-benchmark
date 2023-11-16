import torch
import torch.nn.functional as F

def compute_flash_attention(flash_attn, q, k, v, attention_mask=None, head_mask=None):
    if False:
        while True:
            i = 10
    (batch_size, max_len) = (q.size(0), q.size(1))
    qkv = torch.stack([q, k, v], dim=2)
    dtype_in = qkv.dtype
    if dtype_in == torch.float32:
        qkv = qkv.to(torch.float16)
    (cu_seqlens, max_seqlen) = (None, None)
    if attention_mask is None:
        out = flash_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    else:
        csums = (attention_mask >= 0).cumsum(dim=1)
        ends = csums.argmax(dim=1) + 1
        starts = ends - csums.max(dim=1).values
        seqlens = ends - starts
        qkv = torch.cat([qkv[i, starts[i]:ends[i]] for i in range(batch_size)], dim=0)
        zero = torch.zeros_like(seqlens[:1])
        cu_seqlens = torch.cat([zero, seqlens.cumsum(dim=0)], dim=0).to(torch.int32)
        max_seqlen = seqlens.max().item()
        out = flash_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        seqs = [out[start:end] for (start, end) in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        padded_seqs = [F.pad(seqs[i], (0, 0) * (seqs[i].dim() - 1) + (starts[i], max_len - ends[i]), value=0.0) for i in range(batch_size)]
        out = torch.stack(padded_seqs)
    if out.dtype != dtype_in:
        out = out.to(dtype_in)
    return out
if __name__ == '__main__':
    from flash_attn.modules.mha import FlashSelfAttention
    flash_attn = FlashSelfAttention(causal=True)
    dtype = torch.float16
    device = torch.device('cuda:0')
    (batch_size, seq_len, num_heads, head_size) = (4, 18, 8, 32)
    q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype, device=device)
    attn_mask = torch.randn(batch_size, seq_len, dtype=dtype, device=device).abs().cumsum(dim=1)
    attn_mask = ((attn_mask > 3) & (attn_mask < 10)).int().log()
    out = compute_flash_attention(flash_attn, q, k, v, attention_mask=attn_mask)