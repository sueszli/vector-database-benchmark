from .core import MaskedTensor
__all__ = ['as_masked_tensor', 'masked_tensor']
'"\nThese two factory functions are intended to mirror\n    torch.tensor - guaranteed to be a leaf node\n    torch.as_tensor - differentiable constructor that preserves the autograd history\n'

def masked_tensor(data, mask, requires_grad=False):
    if False:
        for i in range(10):
            print('nop')
    return MaskedTensor(data, mask, requires_grad)

def as_masked_tensor(data, mask):
    if False:
        return 10
    return MaskedTensor._from_values(data, mask)