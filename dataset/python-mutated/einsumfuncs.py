from __future__ import annotations
import numpy as np
from dask.array.core import asarray, blockwise, einsum_lookup
from dask.utils import derived_from
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

def chunk_einsum(*operands, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    subscripts = kwargs.pop('subscripts')
    ncontract_inds = kwargs.pop('ncontract_inds')
    dtype = kwargs.pop('kernel_dtype')
    einsum = einsum_lookup.dispatch(type(operands[0]))
    chunk = einsum(subscripts, *operands, dtype=dtype, **kwargs)
    return chunk.reshape(chunk.shape + (1,) * ncontract_inds)

def parse_einsum_input(operands):
    if False:
        return 10
    "\n    A reproduction of numpy's _parse_einsum_input()\n    which in itself is a reproduction of\n    c side einsum parsing in python.\n\n    Returns\n    -------\n    input_strings : str\n        Parsed input strings\n    output_string : str\n        Parsed output string\n    operands : list of array_like\n        The operands to use in the numpy contraction\n    Examples\n    --------\n    The operand list is simplified to reduce printing:\n    >> a = np.random.rand(4, 4)\n    >> b = np.random.rand(4, 4, 4)\n    >> __parse_einsum_input(('...a,...a->...', a, b))\n    ('za,xza', 'xz', [a, b])\n    >> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))\n    ('za,xza', 'xz', [a, b])\n    "
    if len(operands) == 0:
        raise ValueError('No input operands')
    if isinstance(operands[0], str):
        subscripts = operands[0].replace(' ', '')
        operands = [asarray(o) for o in operands[1:]]
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols_set:
                raise ValueError('Character %s is not a valid symbol.' % s)
    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for _ in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))
        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asarray(v) for v in operand_list]
        subscripts = ''
        last = len(subscript_list) - 1
        for (num, sub) in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += '...'
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError('For this input type lists must contain either int or Ellipsis')
            if num != last:
                subscripts += ','
        if output_list is not None:
            subscripts += '->'
            for s in output_list:
                if s is Ellipsis:
                    subscripts += '...'
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError('For this input type lists must contain either int or Ellipsis')
    if '-' in subscripts or '>' in subscripts:
        invalid = subscripts.count('-') > 1 or subscripts.count('>') > 1
        if invalid or subscripts.count('->') != 1:
            raise ValueError("Subscripts can only contain one '->'.")
    if '.' in subscripts:
        used = subscripts.replace('.', '').replace(',', '').replace('->', '')
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = ''.join(unused)
        longest = 0
        if '->' in subscripts:
            (input_tmp, output_sub) = subscripts.split('->')
            split_subscripts = input_tmp.split(',')
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False
        for (num, sub) in enumerate(split_subscripts):
            if '.' in sub:
                if sub.count('.') != 3 or sub.count('...') != 1:
                    raise ValueError('Invalid Ellipses.')
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(sub) - 3
                if ellipse_count > longest:
                    longest = ellipse_count
                if ellipse_count < 0:
                    raise ValueError('Ellipses lengths do not match.')
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)
        subscripts = ','.join(split_subscripts)
        if longest == 0:
            out_ellipse = ''
        else:
            out_ellipse = ellipse_inds[-longest:]
        if out_sub:
            subscripts += '->' + output_sub.replace('...', out_ellipse)
        else:
            output_subscript = ''
            tmp_subscripts = subscripts.replace(',', '')
            for s in sorted(set(tmp_subscripts)):
                if s not in einsum_symbols_set:
                    raise ValueError('Character %s is not a valid symbol.' % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))
            subscripts += '->' + out_ellipse + normal_inds
    if '->' in subscripts:
        (input_subscripts, output_subscript) = subscripts.split('->')
    else:
        input_subscripts = subscripts
        tmp_subscripts = subscripts.replace(',', '')
        output_subscript = ''
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols_set:
                raise ValueError('Character %s is not a valid symbol.' % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError('Output character %s did not appear in the input' % char)
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError('Number of einsum subscripts must be equal to the number of operands.')
    return (input_subscripts, output_subscript, operands)

@derived_from(np)
def einsum(*operands, dtype=None, optimize=False, split_every=None, **kwargs):
    if False:
        while True:
            i = 10
    'Dask added an additional keyword-only argument ``split_every``.\n\n    split_every: int >= 2 or dict(axis: int), optional\n        Determines the depth of the recursive aggregation.\n        Deafults to ``None`` which would let dask heuristically\n        decide a good default.\n    '
    einsum_dtype = dtype
    (inputs, outputs, ops) = parse_einsum_input(operands)
    subscripts = '->'.join((inputs, outputs))
    if dtype is None:
        dtype = np.result_type(*[o.dtype for o in ops])
    if optimize is not False:
        fake_ops = [np.broadcast_to(o.dtype.type(0), shape=o.shape) for o in ops]
        (optimize, _) = np.einsum_path(subscripts, *fake_ops, optimize=optimize)
    inputs = [tuple(i) for i in inputs.split(',')]
    all_inds = {a for i in inputs for a in i}
    contract_inds = all_inds - set(outputs)
    ncontract_inds = len(contract_inds)
    result = blockwise(chunk_einsum, tuple(outputs) + tuple(contract_inds), *(a for ap in zip(ops, inputs) for a in ap), adjust_chunks={ind: 1 for ind in contract_inds}, dtype=dtype, subscripts=subscripts, kernel_dtype=einsum_dtype, ncontract_inds=ncontract_inds, optimize=optimize, **kwargs)
    if ncontract_inds > 0:
        size = len(outputs)
        return result.sum(axis=list(range(size, size + ncontract_inds)), split_every=split_every)
    return result