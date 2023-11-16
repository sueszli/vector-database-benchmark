""" Wrapper for ngram_repeat_block cuda extension """
import math
import warnings
from typing import List
import torch
from torch import nn
try:
    from fairseq import ngram_repeat_block_cuda
    EXTENSION_BUILT = True
except ImportError:
    EXTENSION_BUILT = False

def is_cuda_extension_usable() -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether ngram_repeat_block_cuda is built properly'
    if not EXTENSION_BUILT or not torch.cuda.is_available():
        return False
    bsz = 2
    tokens = torch.tensor([[4, 4, 3, 2], [1, 2, 3, 4]], dtype=torch.long, device='cuda')
    lprobs = torch.rand((8, 12), device='cuda')
    try:
        outputs = ngram_repeat_block_cuda.forward(tokens, lprobs, bsz, 3, 4, 3)
        outputs = outputs + 4
        return True
    except RuntimeError:
        warnings.warn('NGramRepeatBlock extension must be rebuilt.Run TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0" python setup.py build_ext --inplace')
        return False

class NGramRepeatBlock(nn.Module):
    """Wrapper class for calling ngram_repeat_block cuda extension"""

    def __init__(self, no_repeat_ngram_size: int, use_extension: bool=True):
        if False:
            while True:
                i = 10
        super().__init__()
        self.use_extension = is_cuda_extension_usable() if use_extension else False
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def reset_parameters(self):
        if False:
            while True:
                i = 10
        pass

    @torch.jit.unused
    def call_cuda_extension(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        if False:
            i = 10
            return i + 15
        return ngram_repeat_block_cuda.forward(tokens, lprobs, bsz, step, beam_size, self.no_repeat_ngram_size)

    def forward(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            tokens(Tensor): Input tokens(Bsz*beam, seq_len)\n            lprobs(Tensor): likelihood probability,\n            Expected to be updated in place.(Bsz*beam, vocab_size)\n            bsz(int): batch size\n            step(int): current step\n            beam_size(int): beam size\n            no_repeat_ngram_size(int): Ngram size\n        '
        msg = f'expected {bsz * beam_size} got'
        assert tokens.size(0) == bsz * beam_size, f'{msg} {tokens.size(0)}'
        assert lprobs.size(0) == bsz * beam_size, f'{msg} {lprobs.size(0)}'
        if self.use_extension:
            return self.call_cuda_extension(tokens, lprobs, bsz, beam_size, step)
        else:
            return self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        if False:
            return 10
        'For each hypothesis generate a list of previous ngrams and set associated lprobs to -inf'
        banned_tokens = [torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)]
        if step + 2 - self.no_repeat_ngram_size >= 0:
            cpu_tokens: List[List[int]] = tokens.cpu().tolist()
            check_start_pos = step + 2 - self.no_repeat_ngram_size
            for bbsz_idx in range(bsz * beam_size):
                ngram_to_check = cpu_tokens[bbsz_idx][-(self.no_repeat_ngram_size - 1):]
                for i in range(check_start_pos):
                    if ngram_to_check == cpu_tokens[bbsz_idx][i:i + self.no_repeat_ngram_size - 1]:
                        banned_tokens[bbsz_idx].append(cpu_tokens[bbsz_idx][i + self.no_repeat_ngram_size - 1])
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][torch.tensor(banned_tokens[bbsz_idx], dtype=torch.int64)] = torch.tensor(-math.inf).to(lprobs)
        return lprobs