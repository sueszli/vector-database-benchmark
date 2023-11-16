import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
        if False:
            for i in range(10):
                print('nop')
        nn.Module.__init__(self)
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        if False:
            while True:
                i = 10
        mask = self.mask_template.float() + self.current_val.float() * self._max_size
        mask = mask / self._ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self._max_size:
            mask = mask.narrow(-1, self._max_size - x.size(-1), x.size(-1))
        x = (x * mask).type_as(x)
        return x

    def get_current_max_size(self, include_ramp=True):
        if False:
            return 10
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        if False:
            i = 10
            return i + 15
        current_size = math.ceil(self.current_val.float().mean().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        if False:
            for i in range(10):
                print('nop')
        'this need to be called after each update'
        self.current_val.data.clamp_(0, 1)

class AdaptiveSpan(nn.Module):
    """Adaptive attention span for Transformerself.
    This module learns an attention span length from data for each
    self-attention head.
    Args:
        attn_span: maximum attention span
        adapt_span_loss: loss coefficient for the span length
        adapt_span_ramp: length of the masking ramp
        adapt_span_init: initial size ratio
        adapt_span_cache: adapt cache size to reduce memory usage
    """

    def __init__(self, attn_span, adapt_span_ramp, adapt_span_init, n_head, adapt_span_layer, **kargs):
        if False:
            i = 10
            return i + 15
        nn.Module.__init__(self)
        self._max_span = attn_span
        self._n_head = n_head
        self._adapt_span_layer = adapt_span_layer
        if self._adapt_span_layer:
            self._mask = AdaptiveMask(max_size=self._max_span, ramp_size=adapt_span_ramp, init_val=adapt_span_init)
        else:
            self._mask = AdaptiveMask(max_size=self._max_span, ramp_size=adapt_span_ramp, init_val=adapt_span_init, shape=(n_head, 1, 1))

    def forward(self, attn, normalize=True):
        if False:
            i = 10
            return i + 15
        'mask attention with the right span'
        self.clamp_param()
        if self._adapt_span_layer:
            attn = self._mask(attn)
        else:
            B = attn.size(0)
            M = attn.size(1)
            attn = attn.reshape(B // self._n_head, self._n_head, M, -1)
            attn = self._mask(attn)
            attn = attn.view(B, M, -1)
        return attn

    def get_trim_len(self):
        if False:
            i = 10
            return i + 15
        'how much of memory can be trimmed to reduce computation'
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe):
        if False:
            i = 10
            return i + 15
        'trim out unnecessary memory beforehand to reduce computation'
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self._max_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return (key, value, key_pe)

    def get_cache_size(self):
        if False:
            print('Hello World!')
        'determine how long the cache should be'
        trim_len = self.get_trim_len()
        return min(self._max_span, self._max_span - trim_len + 64)

    def get_loss(self):
        if False:
            while True:
                i = 10
        'a loss term for regularizing the span length'
        return self._max_span * self._mask.current_val.float().mean()

    def get_current_max_span(self):
        if False:
            print('Hello World!')
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        if False:
            while True:
                i = 10
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        if False:
            for i in range(10):
                print('nop')
        self._mask.clamp_param()