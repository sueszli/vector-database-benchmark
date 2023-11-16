from typing import Optional
import torch
from torch import Tensor
from examples.simultaneous_translation.utils.functions import exclusive_cumprod, prob_check, moving_sum

def expected_alignment_from_p_choose(p_choose: Tensor, padding_mask: Optional[Tensor]=None, eps: float=1e-06):
    if False:
        return 10
    '\n    Calculating expected alignment for from stepwise probability\n\n    Reference:\n    Online and Linear-Time Attention by Enforcing Monotonic Alignments\n    https://arxiv.org/pdf/1704.00784.pdf\n\n    q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}\n    a_ij = p_ij q_ij\n\n    Parallel solution:\n    ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))\n\n    ============================================================\n    Expected input size\n    p_choose: bsz, tgt_len, src_len\n    '
    prob_check(p_choose)
    (bsz, tgt_len, src_len) = p_choose.size()
    dtype = p_choose.dtype
    p_choose = p_choose.float()
    if padding_mask is not None:
        p_choose = p_choose.masked_fill(padding_mask.unsqueeze(1), 0.0)
    if p_choose.is_cuda:
        p_choose = p_choose.contiguous()
        from alignment_train_cuda_binding import alignment_train_cuda as alignment_train
    else:
        from alignment_train_cpu_binding import alignment_train_cpu as alignment_train
    alpha = p_choose.new_zeros([bsz, tgt_len, src_len])
    alignment_train(p_choose, alpha, eps)
    alpha = alpha.type(dtype)
    prob_check(alpha)
    return alpha

def expected_soft_attention(alpha: Tensor, soft_energy: Tensor, padding_mask: Optional[Tensor]=None, chunk_size: Optional[int]=None, eps: float=1e-10):
    if False:
        while True:
            i = 10
    '\n    Function to compute expected soft attention for\n    monotonic infinite lookback attention from\n    expected alignment and soft energy.\n\n    Reference:\n    Monotonic Chunkwise Attention\n    https://arxiv.org/abs/1712.05382\n\n    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation\n    https://arxiv.org/abs/1906.05218\n\n    alpha: bsz, tgt_len, src_len\n    soft_energy: bsz, tgt_len, src_len\n    padding_mask: bsz, src_len\n    left_padding: bool\n    '
    if padding_mask is not None:
        alpha = alpha.masked_fill(padding_mask.unsqueeze(1), 0.0)
        soft_energy = soft_energy.masked_fill(padding_mask.unsqueeze(1), -float('inf'))
    prob_check(alpha)
    dtype = alpha.dtype
    alpha = alpha.float()
    soft_energy = soft_energy.float()
    soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
    exp_soft_energy = torch.exp(soft_energy) + eps
    if chunk_size is not None:
        beta = exp_soft_energy * moving_sum(alpha / (eps + moving_sum(exp_soft_energy, chunk_size, 1)), 1, chunk_size)
    else:
        inner_items = alpha / (eps + torch.cumsum(exp_soft_energy, dim=2))
        beta = exp_soft_energy * torch.cumsum(inner_items.flip(dims=[2]), dim=2).flip(dims=[2])
    if padding_mask is not None:
        beta = beta.masked_fill(padding_mask.unsqueeze(1).to(torch.bool), 0.0)
    beta = beta.type(dtype)
    beta = beta.clamp(0, 1)
    prob_check(beta)
    return beta

def mass_preservation(alpha: Tensor, padding_mask: Optional[Tensor]=None, left_padding: bool=False):
    if False:
        return 10
    '\n    Function to compute the mass perservation for alpha.\n    This means that the residual weights of alpha will be assigned\n    to the last token.\n\n    Reference:\n    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation\n    https://arxiv.org/abs/1906.05218\n\n    alpha: bsz, tgt_len, src_len\n    padding_mask: bsz, src_len\n    left_padding: bool\n    '
    prob_check(alpha)
    if padding_mask is not None:
        if not left_padding:
            assert not padding_mask[:, 0].any(), 'Find padding on the beginning of the sequence.'
        alpha = alpha.masked_fill(padding_mask.unsqueeze(1), 0.0)
    if left_padding or padding_mask is None:
        residuals = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0, 1)
        alpha[:, :, -1] = residuals
    else:
        (_, tgt_len, src_len) = alpha.size()
        residuals = 1 - alpha.sum(dim=-1, keepdim=True).clamp(0, 1)
        src_lens = src_len - padding_mask.sum(dim=1, keepdim=True)
        src_lens = src_lens.expand(-1, tgt_len).contiguous()
        residuals += alpha.gather(2, src_lens.unsqueeze(2) - 1)
        alpha = alpha.scatter(2, src_lens.unsqueeze(2) - 1, residuals)
        prob_check(alpha)
    return alpha