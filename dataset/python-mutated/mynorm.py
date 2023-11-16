import numpy as np
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
import scipy.linalg

@register_jitable
def _oneD_norm_2(a):
    if False:
        i = 10
        return i + 15
    val = np.abs(a)
    return np.sqrt(np.sum(val * val))

@overload(scipy.linalg.norm)
def jit_norm(a, ord=None):
    if False:
        while True:
            i = 10
    if isinstance(ord, types.Optional):
        ord = ord.type
    if not isinstance(ord, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("'ord' must be either integer or floating-point")
    if not isinstance(a, types.Array):
        raise TypingError('Only accepts NumPy ndarray')
    if not isinstance(a.dtype, (types.Integer, types.Float)):
        raise TypingError('Only integer and floating point types accepted')
    if not 0 <= a.ndim <= 2:
        raise TypingError('3D and beyond are not allowed')
    elif a.ndim == 0:
        return a.item()
    elif a.ndim == 1:

        def _oneD_norm_x(a, ord=None):
            if False:
                i = 10
                return i + 15
            if ord == 2 or ord is None:
                return _oneD_norm_2(a)
            elif ord == np.inf:
                return np.max(np.abs(a))
            elif ord == -np.inf:
                return np.min(np.abs(a))
            elif ord == 0:
                return np.sum(a != 0)
            elif ord == 1:
                return np.sum(np.abs(a))
            else:
                return np.sum(np.abs(a) ** ord) ** (1.0 / ord)
        return _oneD_norm_x
    elif a.ndim == 2:

        def _two_D_norm_2(a, ord=None):
            if False:
                while True:
                    i = 10
            return _oneD_norm_2(a.ravel())
        return _two_D_norm_2
if __name__ == '__main__':

    @njit
    def use(a, ord=None):
        if False:
            print('Hello World!')
        return scipy.linalg.norm(a, ord)
    a = np.arange(10)
    print(use(a))
    print(scipy.linalg.norm(a))
    b = np.arange(9).reshape((3, 3))
    print(use(b))
    print(scipy.linalg.norm(b))