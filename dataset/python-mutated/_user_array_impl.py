"""
Container class for backward compatibility with NumArray.

The user_array.container class exists for backward compatibility with NumArray
and is not meant to be used in new code. If you need to create an array
container class, we recommend either creating a class that wraps an ndarray
or subclasses ndarray.

"""
from numpy._core import array, asarray, absolute, add, subtract, multiply, divide, remainder, power, left_shift, right_shift, bitwise_and, bitwise_or, bitwise_xor, invert, less, less_equal, not_equal, equal, greater, greater_equal, shape, reshape, arange, sin, sqrt, transpose

class container:
    """
    container(data, dtype=None, copy=True)

    Standard container-class for easy multiple-inheritance.

    Methods
    -------
    copy
    tostring
    byteswap
    astype

    """

    def __init__(self, data, dtype=None, copy=True):
        if False:
            while True:
                i = 10
        self.array = array(data, dtype, copy=copy)

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.ndim > 0:
            return self.__class__.__name__ + repr(self.array)[len('array'):]
        else:
            return self.__class__.__name__ + '(' + repr(self.array) + ')'

    def __array__(self, t=None):
        if False:
            print('Hello World!')
        if t:
            return self.array.astype(t)
        return self.array

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.array)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return self._rc(self.array[index])

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        self.array[index] = asarray(value, self.dtype)

    def __abs__(self):
        if False:
            print('Hello World!')
        return self._rc(absolute(self.array))

    def __neg__(self):
        if False:
            return 10
        return self._rc(-self.array)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self._rc(self.array + asarray(other))
    __radd__ = __add__

    def __iadd__(self, other):
        if False:
            print('Hello World!')
        add(self.array, other, self.array)
        return self

    def __sub__(self, other):
        if False:
            print('Hello World!')
        return self._rc(self.array - asarray(other))

    def __rsub__(self, other):
        if False:
            return 10
        return self._rc(asarray(other) - self.array)

    def __isub__(self, other):
        if False:
            while True:
                i = 10
        subtract(self.array, other, self.array)
        return self

    def __mul__(self, other):
        if False:
            return 10
        return self._rc(multiply(self.array, asarray(other)))
    __rmul__ = __mul__

    def __imul__(self, other):
        if False:
            i = 10
            return i + 15
        multiply(self.array, other, self.array)
        return self

    def __div__(self, other):
        if False:
            while True:
                i = 10
        return self._rc(divide(self.array, asarray(other)))

    def __rdiv__(self, other):
        if False:
            while True:
                i = 10
        return self._rc(divide(asarray(other), self.array))

    def __idiv__(self, other):
        if False:
            while True:
                i = 10
        divide(self.array, other, self.array)
        return self

    def __mod__(self, other):
        if False:
            while True:
                i = 10
        return self._rc(remainder(self.array, other))

    def __rmod__(self, other):
        if False:
            print('Hello World!')
        return self._rc(remainder(other, self.array))

    def __imod__(self, other):
        if False:
            return 10
        remainder(self.array, other, self.array)
        return self

    def __divmod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return (self._rc(divide(self.array, other)), self._rc(remainder(self.array, other)))

    def __rdivmod__(self, other):
        if False:
            while True:
                i = 10
        return (self._rc(divide(other, self.array)), self._rc(remainder(other, self.array)))

    def __pow__(self, other):
        if False:
            return 10
        return self._rc(power(self.array, asarray(other)))

    def __rpow__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._rc(power(asarray(other), self.array))

    def __ipow__(self, other):
        if False:
            for i in range(10):
                print('nop')
        power(self.array, other, self.array)
        return self

    def __lshift__(self, other):
        if False:
            i = 10
            return i + 15
        return self._rc(left_shift(self.array, other))

    def __rshift__(self, other):
        if False:
            i = 10
            return i + 15
        return self._rc(right_shift(self.array, other))

    def __rlshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._rc(left_shift(other, self.array))

    def __rrshift__(self, other):
        if False:
            while True:
                i = 10
        return self._rc(right_shift(other, self.array))

    def __ilshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        left_shift(self.array, other, self.array)
        return self

    def __irshift__(self, other):
        if False:
            i = 10
            return i + 15
        right_shift(self.array, other, self.array)
        return self

    def __and__(self, other):
        if False:
            return 10
        return self._rc(bitwise_and(self.array, other))

    def __rand__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._rc(bitwise_and(other, self.array))

    def __iand__(self, other):
        if False:
            print('Hello World!')
        bitwise_and(self.array, other, self.array)
        return self

    def __xor__(self, other):
        if False:
            print('Hello World!')
        return self._rc(bitwise_xor(self.array, other))

    def __rxor__(self, other):
        if False:
            while True:
                i = 10
        return self._rc(bitwise_xor(other, self.array))

    def __ixor__(self, other):
        if False:
            return 10
        bitwise_xor(self.array, other, self.array)
        return self

    def __or__(self, other):
        if False:
            print('Hello World!')
        return self._rc(bitwise_or(self.array, other))

    def __ror__(self, other):
        if False:
            i = 10
            return i + 15
        return self._rc(bitwise_or(other, self.array))

    def __ior__(self, other):
        if False:
            for i in range(10):
                print('nop')
        bitwise_or(self.array, other, self.array)
        return self

    def __pos__(self):
        if False:
            print('Hello World!')
        return self._rc(self.array)

    def __invert__(self):
        if False:
            print('Hello World!')
        return self._rc(invert(self.array))

    def _scalarfunc(self, func):
        if False:
            i = 10
            return i + 15
        if self.ndim == 0:
            return func(self[0])
        else:
            raise TypeError('only rank-0 arrays can be converted to Python scalars.')

    def __complex__(self):
        if False:
            while True:
                i = 10
        return self._scalarfunc(complex)

    def __float__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._scalarfunc(float)

    def __int__(self):
        if False:
            return 10
        return self._scalarfunc(int)

    def __hex__(self):
        if False:
            while True:
                i = 10
        return self._scalarfunc(hex)

    def __oct__(self):
        if False:
            i = 10
            return i + 15
        return self._scalarfunc(oct)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return self._rc(less(self.array, other))

    def __le__(self, other):
        if False:
            print('Hello World!')
        return self._rc(less_equal(self.array, other))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._rc(equal(self.array, other))

    def __ne__(self, other):
        if False:
            return 10
        return self._rc(not_equal(self.array, other))

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._rc(greater(self.array, other))

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        return self._rc(greater_equal(self.array, other))

    def copy(self):
        if False:
            return 10
        ''
        return self._rc(self.array.copy())

    def tostring(self):
        if False:
            i = 10
            return i + 15
        ''
        return self.array.tostring()

    def tobytes(self):
        if False:
            while True:
                i = 10
        ''
        return self.array.tobytes()

    def byteswap(self):
        if False:
            i = 10
            return i + 15
        ''
        return self._rc(self.array.byteswap())

    def astype(self, typecode):
        if False:
            print('Hello World!')
        ''
        return self._rc(self.array.astype(typecode))

    def _rc(self, a):
        if False:
            while True:
                i = 10
        if len(shape(a)) == 0:
            return a
        else:
            return self.__class__(a)

    def __array_wrap__(self, *args):
        if False:
            print('Hello World!')
        return self.__class__(args[0])

    def __setattr__(self, attr, value):
        if False:
            i = 10
            return i + 15
        if attr == 'array':
            object.__setattr__(self, attr, value)
            return
        try:
            self.array.__setattr__(attr, value)
        except AttributeError:
            object.__setattr__(self, attr, value)

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr == 'array':
            return object.__getattribute__(self, attr)
        return self.array.__getattribute__(attr)
if __name__ == '__main__':
    temp = reshape(arange(10000), (100, 100))
    ua = container(temp)
    print(dir(ua))
    print(shape(ua), ua.shape)
    ua_small = ua[:3, :5]
    print(ua_small)
    ua_small[0, 0] = 10
    print(ua_small[0, 0], ua[0, 0])
    print(sin(ua_small) / 3.0 * 6.0 + sqrt(ua_small ** 2))
    print(less(ua_small, 103), type(less(ua_small, 103)))
    print(type(ua_small * reshape(arange(15), shape(ua_small))))
    print(reshape(ua_small, (5, 3)))
    print(transpose(ua_small))