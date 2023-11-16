import itertools
import numpy
import six
from chainer import testing
import chainer.utils

def pooling_patches(dims, ksize, stride, pad, cover_all):
    if False:
        print('Hello World!')
    'Return tuples of slices that indicate pooling patches.'
    if cover_all:
        xss = itertools.product(*[six.moves.range(-p, d + p - k + s, s) for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)])
    else:
        xss = itertools.product(*[six.moves.range(-p, d + p - k + 1, s) for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)])
    return [tuple((slice(max(x, 0), min(x + k, d)) for (x, d, k) in six.moves.zip(xs, dims, ksize))) for xs in xss]

def shuffled_linspace(shape, dtype):
    if False:
        i = 10
        return i + 15
    size = chainer.utils.size_of_shape(shape)
    x = numpy.random.permutation(size) + numpy.random.uniform(0.3, 0.7, size)
    x = (2 * x / max(1, size) - 1).astype(dtype)
    return x.reshape(shape)
testing.run_module(__name__, __file__)