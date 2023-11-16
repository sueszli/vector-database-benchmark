from vyper import ast as vy_ast
from vyper.abi_types import ABI_Bytes, ABI_String, ABIType
from vyper.exceptions import CompilerPanic, StructureException, UnexpectedNodeType, UnexpectedValue
from vyper.semantics.types.base import VyperType
from vyper.semantics.types.utils import get_index_value
from vyper.utils import ceil32

class _BytestringT(VyperType):
    """
    Private base class for single-value types which occupy multiple memory slots
    and where a maximum length must be given via a subscript (string, bytes).

    Types for literals have an inferred minimum length. For example, `b"hello"`
    has a length of 5 of more and so can be used in an operation with `bytes[5]`
    or `bytes[10]`, but not `bytes[4]`. Upon comparison to a fixed length type,
    the minimum length is discarded and the type assumes the fixed length it was
    compared against.

    Attributes
    ----------
    _length : int
        The maximum allowable length of the data within the type.
    _min_length: int
        The minimum length of the data within the type. Used when the type
        is applied to a literal definition.
    """
    _as_darray = True
    _as_hashmap_key = True
    _equality_attrs = ('_length', '_min_length')
    _is_bytestring: bool = True

    def __init__(self, length: int=0) -> None:
        if False:
            return 10
        super().__init__()
        self._length = length
        self._min_length = length

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'{self._id}[{self.length}]'

    @property
    def length(self):
        if False:
            while True:
                i = 10
        '\n        Property method used to check the length of a type.\n        '
        if self._length:
            return self._length
        return self._min_length

    @property
    def maxlen(self):
        if False:
            return 10
        '\n        Alias for backwards compatibility.\n        '
        return self.length

    def validate_literal(self, node: vy_ast.Constant) -> None:
        if False:
            i = 10
            return i + 15
        super().validate_literal(node)
        if len(node.value) != self.length:
            raise CompilerPanic('unreachable')

    @property
    def size_in_bytes(self):
        if False:
            print('Hello World!')
        return 32 + ceil32(self.length)

    def set_length(self, length):
        if False:
            while True:
                i = 10
        '\n        Sets the exact length of the type.\n\n        May only be called once, and only on a type that does not yet have\n        a fixed length.\n        '
        if self._length:
            raise CompilerPanic('Type already has a fixed length')
        self._length = length
        self._min_length = length

    def set_min_length(self, min_length):
        if False:
            print('Hello World!')
        '\n        Sets the minimum length of the type.\n\n        May only be used to increase the minimum length. May not be called if\n        an exact length has been set.\n        '
        if self._length:
            raise CompilerPanic('Type already has a fixed length')
        if self._min_length > min_length:
            raise CompilerPanic('Cannot reduce the min_length of ArrayValueType')
        self._min_length = min_length

    def compare_type(self, other):
        if False:
            i = 10
            return i + 15
        if not super().compare_type(other):
            return False
        if not self._length and (not other._length):
            min_length = max(self._min_length, other._min_length)
            self.set_min_length(min_length)
            other.set_min_length(min_length)
            return True
        if self._length:
            if not other._length:
                other.set_length(max(self._length, other._min_length))
            return self._length >= other._length
        return other.compare_type(self)

    @classmethod
    def from_annotation(cls, node: vy_ast.VyperNode) -> '_BytestringT':
        if False:
            while True:
                i = 10
        if not isinstance(node, vy_ast.Subscript) or not isinstance(node.slice, vy_ast.Index):
            raise StructureException(f'Cannot declare {cls._id} type without a maximum length, e.g. {cls._id}[5]', node)
        if node.get('value.id') != cls._id:
            raise UnexpectedValue('Node id does not match type name')
        length = get_index_value(node.slice)
        return cls(length)

    @classmethod
    def from_literal(cls, node: vy_ast.Constant) -> '_BytestringT':
        if False:
            return 10
        if not isinstance(node, cls._valid_literal):
            raise UnexpectedNodeType(f'Not a {cls._id}: {node}')
        t = cls()
        t.set_min_length(len(node.value))
        return t

class BytesT(_BytestringT):
    _id = 'Bytes'
    _valid_literal = (vy_ast.Bytes,)

    @property
    def abi_type(self) -> ABIType:
        if False:
            i = 10
            return i + 15
        return ABI_Bytes(self.length)

class StringT(_BytestringT):
    _id = 'String'
    _valid_literal = (vy_ast.Str,)

    @property
    def abi_type(self) -> ABIType:
        if False:
            return 10
        return ABI_String(self.length)