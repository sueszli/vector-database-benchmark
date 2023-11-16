import math
import cupy
from cupyx.scipy import special

def _normalize(x, axis):
    if False:
        i = 10
        return i + 15
    'Normalize, preserving floating point precision of x.'
    x_sum = x.sum(axis=axis, keepdims=True)
    if x.dtype.kind == 'f':
        x /= x_sum
    else:
        x = x / x_sum
    return x

def entropy(pk, qk=None, base=None, axis=0):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the entropy of a distribution for given probability values.\n\n    If only probabilities ``pk`` are given, the entropy is calculated as\n    ``S = -sum(pk * log(pk), axis=axis)``.\n\n    If ``qk`` is not None, then compute the Kullback-Leibler divergence\n    ``S = sum(pk * log(pk / qk), axis=axis)``.\n\n    This routine will normalize ``pk`` and ``qk`` if they don't sum to 1.\n\n    Args:\n        pk (ndarray): Defines the (discrete) distribution. ``pk[i]`` is the\n            (possibly unnormalized) probability of event ``i``.\n        qk (ndarray, optional): Sequence against which the relative entropy is\n            computed. Should be in the same format as ``pk``.\n        base (float, optional): The logarithmic base to use, defaults to ``e``\n            (natural logarithm).\n        axis (int, optional): The axis along which the entropy is calculated.\n            Default is 0.\n\n    Returns:\n        S (cupy.ndarray): The calculated entropy.\n\n    "
    if pk.dtype.kind == 'c' or (qk is not None and qk.dtype.kind == 'c'):
        raise TypeError('complex dtype not supported')
    float_type = cupy.float32 if pk.dtype.char in 'ef' else cupy.float64
    pk = pk.astype(float_type, copy=False)
    pk = _normalize(pk, axis)
    if qk is None:
        vec = special.entr(pk)
    else:
        if qk.shape != pk.shape:
            raise ValueError('qk and pk must have same shape.')
        qk = qk.astype(float_type, copy=False)
        qk = _normalize(qk, axis)
        vec = special.rel_entr(pk, qk)
    s = cupy.sum(vec, axis=axis)
    if base is not None:
        s /= math.log(base)
    return s