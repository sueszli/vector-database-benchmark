from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
if TYPE_CHECKING:
    from cupyx.jit._internal_types import Data

class TypeBase:

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def declvar(self, x: str, init: Optional['Data']) -> str:
        if False:
            for i in range(10):
                print('nop')
        if init is None:
            return f'{self} {x}'
        return f'{self} {x} = {init.code}'

    def assign(self, var: 'Data', value: 'Data') -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{var.code} = {value.code}'

class Void(TypeBase):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pass

    def __str__(self) -> str:
        if False:
            return 10
        return 'void'

class Unknown(TypeBase):

    def __init__(self, *, label: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.label = label

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        raise TypeError('unknown type can be used only in ary of a function.')

class Scalar(TypeBase):

    def __init__(self, dtype: npt.DTypeLike) -> None:
        if False:
            print('Hello World!')
        self.dtype = numpy.dtype(dtype)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        dtype = self.dtype
        if dtype == numpy.float16:
            dtype = numpy.dtype('float32')
        return get_typename(dtype)

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        assert isinstance(other, TypeBase)
        return isinstance(other, Scalar) and self.dtype == other.dtype

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self.dtype)

class PtrDiff(Scalar):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__('q')

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return 'ptrdiff_t'

class ArrayBase(TypeBase):

    def ndim(self, instance: 'Data'):
        if False:
            while True:
                i = 10
        from cupyx.jit import _internal_types
        return _internal_types.Constant(self._ndim)

    def __init__(self, child_type: TypeBase, ndim: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(child_type, TypeBase)
        self.child_type = child_type
        self._ndim = ndim

class PointerBase(ArrayBase):

    def __init__(self, child_type: TypeBase) -> None:
        if False:
            while True:
                i = 10
        super().__init__(child_type, 1)

    @staticmethod
    def _add(env, x: 'Data', y: 'Data') -> 'Data':
        if False:
            for i in range(10):
                print('nop')
        from cupyx.jit import _internal_types
        if isinstance(y.ctype, Scalar) and y.ctype.dtype.kind in 'iu':
            return _internal_types.Data(f'({x.code} + {y.code})', x.ctype)
        return NotImplemented

    @staticmethod
    def _radd(env, x: 'Data', y: 'Data') -> 'Data':
        if False:
            i = 10
            return i + 15
        from cupyx.jit import _internal_types
        if isinstance(x.ctype, Scalar) and x.ctype.dtype.kind in 'iu':
            return _internal_types.Data(f'({x.code} + {y.code})', y.ctype)
        return NotImplemented

    @staticmethod
    def _sub(env, x: 'Data', y: 'Data') -> 'Data':
        if False:
            for i in range(10):
                print('nop')
        from cupyx.jit import _internal_types
        if isinstance(y.ctype, Scalar) and y.ctype.dtype.kind in 'iu':
            return _internal_types.Data(f'({x.code} - {y.code})', x.ctype)
        if x.ctype == y.ctype:
            return _internal_types.Data(f'({x.code} - {y.code})', PtrDiff())
        return NotImplemented

class CArray(ArrayBase):
    from cupyx.jit import _internal_types

    def __init__(self, dtype: npt.DTypeLike, ndim: int, is_c_contiguous: bool, index_32_bits: bool) -> None:
        if False:
            while True:
                i = 10
        self.dtype = numpy.dtype(dtype)
        self._ndim = ndim
        self._c_contiguous = is_c_contiguous
        self._index_32_bits = index_32_bits
        super().__init__(Scalar(dtype), ndim)

    @classmethod
    def from_ndarray(cls, x: cupy.ndarray) -> 'CArray':
        if False:
            print('Hello World!')
        return CArray(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)

    def size(self, instance: 'Data') -> 'Data':
        if False:
            for i in range(10):
                print('nop')
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'static_cast<long long>({instance.code}.size())', Scalar('q'))

    def shape(self, instance: 'Data') -> 'Data':
        if False:
            print('Hello World!')
        from cupyx.jit import _internal_types
        if self._ndim > 10:
            raise NotImplementedError('getting shape/strides for an array with ndim > 10 is not supported yet')
        return _internal_types.Data(f'{instance.code}.get_shape()', Tuple([PtrDiff()] * self._ndim))

    def strides(self, instance: 'Data') -> 'Data':
        if False:
            i = 10
            return i + 15
        from cupyx.jit import _internal_types
        if self._ndim > 10:
            raise NotImplementedError('getting shape/strides for an array with ndim > 10 is not supported yet')
        return _internal_types.Data(f'{instance.code}.get_strides()', Tuple([PtrDiff()] * self._ndim))

    @_internal_types.wraps_class_method
    def begin(self, env, instance: 'Data', *args) -> 'Data':
        if False:
            for i in range(10):
                print('nop')
        from cupyx.jit import _internal_types
        if self._ndim != 1:
            raise NotImplementedError('getting begin iterator for an array with ndim != 1 is not supported yet')
        method_name = 'begin_ptr' if self._c_contiguous else 'begin'
        return _internal_types.Data(f'{instance.code}.{method_name}()', CArrayIterator(instance.ctype))

    @_internal_types.wraps_class_method
    def end(self, env, instance: 'Data', *args) -> 'Data':
        if False:
            print('Hello World!')
        from cupyx.jit import _internal_types
        if self._ndim != 1:
            raise NotImplementedError('getting end iterator for an array with ndim != 1 is not supported yet')
        method_name = 'end_ptr' if self._c_contiguous else 'end'
        return _internal_types.Data(f'{instance.code}.{method_name}()', CArrayIterator(instance.ctype))

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        ctype = get_typename(self.dtype)
        ndim = self._ndim
        c_contiguous = get_cuda_code_from_constant(self._c_contiguous, bool_)
        index_32_bits = get_cuda_code_from_constant(self._index_32_bits, bool_)
        return f'CArray<{ctype}, {ndim}, {c_contiguous}, {index_32_bits}>'

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return str(self) == str(other)

    def __hash__(self) -> int:
        if False:
            return 10
        return hash(str(self))

class CArrayIterator(PointerBase):

    def __init__(self, carray_type: CArray) -> None:
        if False:
            i = 10
            return i + 15
        self._carray_type = carray_type
        super().__init__(Scalar(carray_type.dtype))

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{str(self._carray_type)}::iterator'

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        assert isinstance(other, TypeBase)
        return isinstance(other, CArrayIterator) and self._carray_type == other._carray_type

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash((self.dtype, self.ndim, self._c_contiguous, self._index_32_bits))

class SharedMem(ArrayBase):

    def __init__(self, child_type: TypeBase, size: Optional[int], alignment: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        if not (isinstance(size, int) or size is None):
            raise 'size of shared_memory must be integer or `None`'
        if not (isinstance(alignment, int) or alignment is None):
            raise 'alignment must be integer or `None`'
        self._size = size
        self._alignment = alignment
        super().__init__(child_type, 1)

    def declvar(self, x: str, init: Optional['Data']) -> str:
        if False:
            return 10
        assert init is None
        if self._alignment is not None:
            code = f'__align__({self._alignment})'
        else:
            code = ''
        if self._size is None:
            code = f'extern {code} __shared__ {self.child_type} {x}[]'
        else:
            code = f'{code} __shared__ {self.child_type} {x}[{self._size}]'
        return code

class Ptr(PointerBase):

    def __init__(self, child_type: TypeBase) -> None:
        if False:
            while True:
                i = 10
        super().__init__(child_type)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.child_type}*'

class Tuple(TypeBase):

    def __init__(self, types: Sequence[TypeBase]) -> None:
        if False:
            while True:
                i = 10
        self.types = types

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        types = ', '.join([str(t) for t in self.types])
        if len(self.types) == 2:
            return f'STD::pair<{types}>'
        else:
            return f'STD::tuple<{types}>'

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        assert isinstance(other, TypeBase)
        return isinstance(other, Tuple) and self.types == other.types
void: Void = Void()
bool_: Scalar = Scalar(numpy.bool_)
int32: Scalar = Scalar(numpy.int32)
uint32: Scalar = Scalar(numpy.uint32)
uint64: Scalar = Scalar(numpy.uint64)

class Dim3(TypeBase):
    """
    An integer vector type based on uint3 that is used to specify dimensions.

    Attributes:
        x (uint32)
        y (uint32)
        z (uint32)
    """

    def x(self, instance: 'Data') -> 'Data':
        if False:
            return 10
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'{instance.code}.x', uint32)

    def y(self, instance: 'Data') -> 'Data':
        if False:
            print('Hello World!')
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'{instance.code}.y', uint32)

    def z(self, instance: 'Data') -> 'Data':
        if False:
            print('Hello World!')
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'{instance.code}.z', uint32)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return 'dim3'
dim3: Dim3 = Dim3()
_suffix_literals_dict: Mapping[str, str] = {'float64': '', 'float32': 'f', 'int64': 'll', 'int32': '', 'uint64': 'ull', 'uint32': 'u', 'bool': ''}

def get_cuda_code_from_constant(x: Union[bool, int, float, complex], ctype: Scalar) -> str:
    if False:
        while True:
            i = 10
    dtype = ctype.dtype
    suffix_literal = _suffix_literals_dict.get(dtype.name)
    if suffix_literal is not None:
        s = str(x).lower()
        return f'{s}{suffix_literal}'
    ctype_str = str(ctype)
    if dtype.kind == 'c':
        return f'{ctype_str}({x.real}, {x.imag})'
    if ' ' in ctype_str:
        return f'({ctype_str}){x}'
    return f'{ctype_str}({x})'