from typing import Optional, Dict
from torch import Tensor
import torch

def waitk_p_choose(tgt_len: int, src_len: int, bsz: int, waitk_lagging: int, key_padding_mask: Optional[Tensor]=None, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None):
    if False:
        print('Hello World!')
    max_src_len = src_len
    if incremental_state is not None:
        max_tgt_len = incremental_state['steps']['tgt']
        assert max_tgt_len is not None
        max_tgt_len = int(max_tgt_len)
    else:
        max_tgt_len = tgt_len
    if max_src_len < waitk_lagging:
        if incremental_state is not None:
            max_tgt_len = 1
        return torch.zeros(bsz, max_tgt_len, max_src_len)
    activate_indices_offset = (torch.arange(max_tgt_len) * (max_src_len + 1) + waitk_lagging - 1).unsqueeze(0).expand(bsz, max_tgt_len).long()
    if key_padding_mask is not None:
        if key_padding_mask[:, 0].any():
            activate_indices_offset += key_padding_mask.sum(dim=1, keepdim=True)
    activate_indices_offset = activate_indices_offset.clamp(0, min([max_tgt_len, max_src_len - waitk_lagging + 1]) * max_src_len - 1)
    p_choose = torch.zeros(bsz, max_tgt_len * max_src_len)
    p_choose = p_choose.scatter(1, activate_indices_offset, 1.0).view(bsz, max_tgt_len, max_src_len)
    if key_padding_mask is not None:
        p_choose = p_choose.to(key_padding_mask)
        p_choose = p_choose.masked_fill(key_padding_mask.unsqueeze(1), 0)
    if incremental_state is not None:
        p_choose = p_choose[:, -1:]
    return p_choose.float()

def learnable_p_choose(energy, noise_mean: float=0.0, noise_var: float=0.0, training: bool=True):
    if False:
        i = 10
        return i + 15
    '\n    Calculating step wise prob for reading and writing\n    1 to read, 0 to write\n    energy: bsz, tgt_len, src_len\n    '
    noise = 0
    if training:
        noise = torch.normal(noise_mean, noise_var, energy.size()).type_as(energy).to(energy.device)
    p_choose = torch.sigmoid(energy + noise)
    return p_choose