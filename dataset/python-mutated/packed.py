import math
import torch
from pyro.distributions.util import is_identically_one
from pyro.util import ignore_jit_warnings

def pack(value, dim_to_symbol):
    if False:
        print('Hello World!')
    '\n    Converts an unpacked tensor to a packed tensor.\n\n    :param value: a number or tensor\n    :param dim_to_symbol: a map from negative integers to characters\n    '
    if isinstance(value, torch.Tensor):
        assert not hasattr(value, '_pyro_dims'), 'tried to pack an already-packed tensor'
        shape = value.shape
        shift = len(shape)
        try:
            with ignore_jit_warnings():
                dims = ''.join((dim_to_symbol[dim - shift] for (dim, size) in enumerate(shape) if size > 1))
        except KeyError as e:
            raise ValueError('\n  '.join(['Invalid tensor shape.', 'Allowed dims: {}'.format(', '.join(map(str, sorted(dim_to_symbol)))), 'Actual shape: {}'.format(tuple(value.shape)), "Try adding shape assertions for your model's sample values and distribution parameters."])) from e
        value = value.squeeze()
        value._pyro_dims = dims
        assert value.dim() == len(value._pyro_dims)
    return value

def unpack(value, symbol_to_dim):
    if False:
        print('Hello World!')
    '\n    Converts a packed tensor to an unpacked tensor.\n\n    :param value: a number or tensor\n    :param symbol_to_dim: a map from characters to negative integers\n    '
    if isinstance(value, torch.Tensor):
        assert value.dim() == len(value._pyro_dims)
        if value.dim():
            unsorted_dims = [symbol_to_dim[dim] for dim in value._pyro_dims]
            dims = sorted(unsorted_dims)
            value = value.permute(*(unsorted_dims.index(dim) for dim in dims))
            shape = [1] * -min(dims)
            for (dim, size) in zip(dims, value.shape):
                shape[dim] = size
            value = value.reshape(shape)
        else:
            value = value[...]
    return value

def broadcast_all(*values, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Packed broadcasting of multiple tensors.\n    '
    dims = kwargs.get('dims')
    sizes = {dim: size for value in values for (dim, size) in zip(value._pyro_dims, value.shape)}
    if dims is None:
        dims = ''.join(sorted(sizes))
    else:
        assert set(dims) == set(sizes)
    shape = torch.Size((sizes[dim] for dim in dims))
    values = list(values)
    for (i, x) in enumerate(values):
        old_dims = x._pyro_dims
        if old_dims != dims:
            x = x.permute(tuple((old_dims.index(dim) for dim in dims if dim in old_dims)))
            x = x.reshape(tuple((sizes[dim] if dim in old_dims else 1 for dim in dims)))
            x = x.expand(shape)
            x._pyro_dims = dims
            assert x.dim() == len(x._pyro_dims)
            values[i] = x
    return tuple(values)

def gather(value, index, dim):
    if False:
        while True:
            i = 10
    '\n    Packed broadcasted gather of indexed values along a named dim.\n    '
    assert dim in value._pyro_dims
    assert dim not in index._pyro_dims
    (value, index) = broadcast_all(value, index)
    dims = value._pyro_dims.replace(dim, '')
    pos = value._pyro_dims.index(dim)
    with ignore_jit_warnings():
        zero = torch.zeros(1, dtype=torch.long, device=index.device)
    index = index.index_select(pos, zero)
    value = value.gather(pos, index).squeeze(pos)
    value._pyro_dims = dims
    assert value.dim() == len(value._pyro_dims)
    return value

def mul(lhs, rhs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Packed broadcasted multiplication.\n    '
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        dims = ''.join(sorted(set(lhs._pyro_dims + rhs._pyro_dims)))
        equation = lhs._pyro_dims + ',' + rhs._pyro_dims + '->' + dims
        result = torch.einsum(equation, lhs, rhs, backend='torch')
        result._pyro_dims = dims
        return result
    result = lhs * rhs
    if isinstance(lhs, torch.Tensor):
        result._pyro_dims = lhs._pyro_dims
    elif isinstance(rhs, torch.Tensor):
        result._pyro_dims = rhs._pyro_dims
    return result

def scale_and_mask(tensor, scale=1.0, mask=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Scale and mask a packed tensor, broadcasting and avoiding unnecessary ops.\n\n    :param torch.Tensor tensor: a packed tensor\n    :param scale: a positive scale\n    :type scale: torch.Tensor or number\n    :param mask: an optional packed tensor mask\n    :type mask: torch.BoolTensor, bool, or None\n    '
    if isinstance(scale, torch.Tensor) and scale.dim():
        raise NotImplementedError('non-scalar scale is not supported')
    if mask is None or mask is True:
        if is_identically_one(scale):
            return tensor
        result = tensor * scale
        result._pyro_dims = tensor._pyro_dims
        return result
    if mask is False:
        result = torch.zeros_like(tensor)
        result._pyro_dims = tensor._pyro_dims
        return result
    (tensor, mask) = broadcast_all(tensor, mask)
    result = torch.where(mask, tensor, tensor.new_zeros(()))
    result._pyro_dims = tensor._pyro_dims
    return result

def neg(value):
    if False:
        i = 10
        return i + 15
    '\n    Packed negation.\n    '
    result = -value
    if isinstance(value, torch.Tensor):
        result._pyro_dims = value._pyro_dims
    return result

def exp(value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Packed pointwise exponential.\n    '
    if isinstance(value, torch.Tensor):
        result = value.exp()
        result._pyro_dims = value._pyro_dims
    else:
        result = math.exp(value)
    return result

def rename_equation(equation, *operands):
    if False:
        for i in range(10):
            print('nop')
    '\n    Renames symbols in an einsum/ubersum equation to match the\n    ``.pyro_dims`` attributes of packed ``operands``.\n    '
    (inputs, outputs) = equation.split('->')
    inputs = inputs.split(',')
    assert len(inputs) == len(operands)
    rename = {old: new for (input_, operand) in zip(inputs, operands) for (old, new) in zip(input_, operand._pyro_dims)}
    return ''.join((rename.get(s, s) for s in equation))