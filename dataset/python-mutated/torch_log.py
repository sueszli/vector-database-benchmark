import torch
from pyro.ops.einsum.util import Tensordot
from pyro.ops.special import safe_log

def transpose(a, axes):
    if False:
        print('Hello World!')
    return a.permute(axes)

def einsum(equation, *operands):
    if False:
        for i in range(10):
            print('nop')
    '\n    Log-sum-exp implementation of einsum.\n    '
    symbols = sorted(set(equation) - set(',->'))
    rename = dict(zip(symbols, 'abcdefghijklmnopqrstuvwxyz'))
    equation = ''.join((rename.get(s, s) for s in equation))
    (inputs, output) = equation.split('->')
    if inputs == output:
        return operands[0].clone()
    inputs = inputs.split(',')
    shifts = []
    exp_operands = []
    for (dims, operand) in zip(inputs, operands):
        shift = operand.detach()
        for (i, dim) in enumerate(dims):
            if dim not in output:
                shift = shift.max(i, keepdim=True)[0]
        shift = shift.clamp(min=torch.finfo(shift.dtype).min)
        exp_operands.append((operand - shift).exp())
        shift = shift.reshape(torch.Size((size for (size, dim) in zip(operand.shape, dims) if dim in output)))
        if shift.dim():
            shift = shift.reshape((1,) * (len(output) - shift.dim()) + shift.shape)
            dims = [dim for dim in dims if dim in output]
            dims = [dim for dim in output if dim not in dims] + dims
            shift = shift.permute(*(dims.index(dim) for dim in output))
        shifts.append(shift)
    result = safe_log(torch.einsum(equation, exp_operands))
    return sum(shifts + [result])
tensordot = Tensordot(einsum)
__all__ = ['transpose', 'einsum', 'tensordot']