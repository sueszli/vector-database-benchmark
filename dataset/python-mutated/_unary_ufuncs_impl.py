"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `_numpy/_ufuncs.py` module.
"""
import torch
from torch import absolute as fabs, arccos, arccosh, arcsin, arcsinh, arctan, arctanh, bitwise_not, bitwise_not as invert, ceil, conj_physical as conjugate, cos, cosh, deg2rad, deg2rad as radians, exp, exp2, expm1, floor, isfinite, isinf, isnan, log, log10, log1p, log2, logical_not, negative, rad2deg, rad2deg as degrees, reciprocal, round as fix, round as rint, sign, signbit, sin, sinh, sqrt, square, tan, tanh, trunc

def cbrt(x):
    if False:
        i = 10
        return i + 15
    return torch.pow(x, 1 / 3)

def positive(x):
    if False:
        print('Hello World!')
    return +x

def absolute(x):
    if False:
        for i in range(10):
            print('nop')
    if x.dtype == torch.bool:
        return x
    return torch.absolute(x)
abs = absolute
conj = conjugate