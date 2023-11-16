import torch
__all__ = ['Dropout']

class Dropout(torch.nn.Dropout):
    """This is the quantized equivalent of :class:`~torch.nn.Dropout`.
        And this is a placeholder to enable models where fp32 tensors
        had dropout to work with quantized tensors in train and eval mode.

    Args:
        p: probability of an element to be zeroed
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        return input

    def _get_name(self):
        if False:
            while True:
                i = 10
        return 'QuantizedDropout'

    @classmethod
    def from_float(cls, mod):
        if False:
            for i in range(10):
                print('nop')
        return cls(mod.p, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        if False:
            while True:
                i = 10
        return cls(mod.p, mod.inplace)