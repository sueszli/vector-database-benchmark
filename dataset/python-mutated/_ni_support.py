from collections.abc import Iterable
import operator
import warnings
import numpy

def _extend_mode_to_code(mode):
    if False:
        while True:
            i = 10
    'Convert an extension mode to the corresponding integer code.\n    '
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode in ['reflect', 'grid-mirror']:
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    elif mode == 'grid-wrap':
        return 5
    elif mode == 'grid-constant':
        return 6
    else:
        raise RuntimeError('boundary mode not supported')

def _normalize_sequence(input, rank):
    if False:
        return 10
    'If input is a scalar, create a sequence of length equal to the\n    rank by duplicating the input. If input is a sequence,\n    check if its length is equal to the length of array.\n    '
    is_str = isinstance(input, str)
    if not is_str and isinstance(input, Iterable):
        normalized = list(input)
        if len(normalized) != rank:
            err = 'sequence argument must have length equal to input rank'
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized

def _get_output(output, input, shape=None, complex_output=False):
    if False:
        return 10
    if shape is None:
        shape = input.shape
    if output is None:
        if not complex_output:
            output = numpy.zeros(shape, dtype=input.dtype.name)
        else:
            complex_type = numpy.promote_types(input.dtype, numpy.complex64)
            output = numpy.zeros(shape, dtype=complex_type)
    elif isinstance(output, (type, numpy.dtype)):
        if complex_output and numpy.dtype(output).kind != 'c':
            warnings.warn('promoting specified output dtype to complex')
            output = numpy.promote_types(output, numpy.complex64)
        output = numpy.zeros(shape, dtype=output)
    elif isinstance(output, str):
        f_dict = {'f': numpy.float32, 'd': numpy.float64, 'F': numpy.complex64, 'D': numpy.complex128}
        output = f_dict[output]
        if complex_output and numpy.dtype(output).kind != 'c':
            raise RuntimeError('output must have complex dtype')
        output = numpy.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError('output shape not correct')
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError('output must have complex dtype')
    return output

def _check_axes(axes, ndim):
    if False:
        for i in range(10):
            print('nop')
    if axes is None:
        return tuple(range(ndim))
    elif numpy.isscalar(axes):
        axes = (operator.index(axes),)
    elif isinstance(axes, Iterable):
        for ax in axes:
            axes = tuple((operator.index(ax) for ax in axes))
            if ax < -ndim or ax > ndim - 1:
                raise ValueError(f'specified axis: {ax} is out of range')
        axes = tuple((ax % ndim if ax < 0 else ax for ax in axes))
    else:
        message = 'axes must be an integer, iterable of integers, or None'
        raise ValueError(message)
    if len(tuple(set(axes))) != len(axes):
        raise ValueError('axes must be unique')
    return axes