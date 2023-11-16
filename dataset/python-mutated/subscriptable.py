import warnings
from typing import Any, Dict, Optional, Tuple, Union
from vyper import ast as vy_ast
from vyper.abi_types import ABI_DynamicArray, ABI_StaticArray, ABI_Tuple, ABIType
from vyper.exceptions import ArrayIndexException, InvalidType, StructureException
from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types.base import VyperType
from vyper.semantics.types.primitives import IntegerT
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.semantics.types.utils import get_index_value, type_from_annotation

class _SubscriptableT(VyperType):
    """
    Base class for subscriptable types such as arrays and mappings.

    Attributes
    ----------
    key_type: VyperType
        Type representing the index for this object.
    value_type : VyperType
        Type representing the value(s) contained in this object.
    """

    def __init__(self, key_type: VyperType, value_type: VyperType) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.key_type = key_type
        self.value_type = value_type

    @property
    def getter_signature(self) -> Tuple[Tuple, Optional[VyperType]]:
        if False:
            print('Hello World!')
        (child_keys, return_type) = self.value_type.getter_signature
        return ((self.key_type,) + child_keys, return_type)

    def validate_index_type(self, node):
        if False:
            for i in range(10):
                print('nop')
        from vyper.semantics.analysis.utils import validate_expected_type
        validate_expected_type(node, self.key_type)

class HashMapT(_SubscriptableT):
    _id = 'HashMap'
    _equality_attrs = ('key_type', 'value_type')
    _invalid_locations = (DataLocation.UNSET, DataLocation.CALLDATA, DataLocation.CODE, DataLocation.MEMORY)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'HashMap[{self.key_type}, {self.value_type}]'

    def compare_type(self, other):
        if False:
            while True:
                i = 10
        return super().compare_type(other) and self.key_type == other.key_type and (self.value_type == other.value_type)

    def get_subscripted_type(self, node):
        if False:
            print('Hello World!')
        return self.value_type

    @classmethod
    def from_annotation(cls, node: Union[vy_ast.Name, vy_ast.Call, vy_ast.Subscript]) -> 'HashMapT':
        if False:
            while True:
                i = 10
        if not isinstance(node, vy_ast.Subscript) or not isinstance(node.slice, vy_ast.Index) or (not isinstance(node.slice.value, vy_ast.Tuple)) or (len(node.slice.value.elements) != 2):
            raise StructureException('HashMap must be defined with a key type and a value type, e.g. my_hashmap: HashMap[k, v]', node)
        (k_ast, v_ast) = node.slice.value.elements
        key_type = type_from_annotation(k_ast, DataLocation.STORAGE)
        if not key_type._as_hashmap_key:
            raise InvalidType('can only use primitive types as HashMap key!', k_ast)
        value_type = type_from_annotation(v_ast, DataLocation.STORAGE)
        return cls(key_type, value_type)

class _SequenceT(_SubscriptableT):
    """
    Private base class for sequence types (i.e., index is an int)

    Arguments
    ---------
    length : int
        Number of items in the type.
    """
    _equality_attrs: tuple = ('value_type', 'length')
    _is_array_type: bool = True

    def __init__(self, value_type: VyperType, length: int):
        if False:
            for i in range(10):
                print('nop')
        if not 0 < length < 2 ** 256:
            raise InvalidType('Array length is invalid')
        if length >= 2 ** 64:
            warnings.warn('Use of large arrays can be unsafe!')
        super().__init__(UINT256_T, value_type)
        self.length = length

    @property
    def count(self):
        if False:
            i = 10
            return i + 15
        '\n        Alias for API compatibility\n        '
        return self.length

    def validate_index_type(self, node):
        if False:
            print('Hello World!')
        from vyper.semantics.analysis.utils import validate_expected_type
        if isinstance(node, vy_ast.Int):
            if node.value < 0:
                raise ArrayIndexException('Vyper does not support negative indexing', node)
            if node.value >= self.length:
                raise ArrayIndexException('Index out of range', node)
        validate_expected_type(node, IntegerT.any())

    def get_subscripted_type(self, node):
        if False:
            return 10
        return self.value_type

def _set_first_key(xs: Dict[str, Any], k: str, val: Any) -> dict:
    if False:
        return 10
    xs.pop(k, None)
    return {k: val, **xs}

class SArrayT(_SequenceT):
    """
    Static array type
    """

    def __init__(self, value_type: VyperType, length: int) -> None:
        if False:
            return 10
        super().__init__(value_type, length)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.value_type}[{self.length}]'

    @property
    def _as_array(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value_type._as_array

    @property
    def abi_type(self) -> ABIType:
        if False:
            return 10
        return ABI_StaticArray(self.value_type.abi_type, self.length)

    def to_abi_arg(self, name: str='') -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        ret = self.value_type.to_abi_arg()
        ret['type'] += f'[{self.length}]'
        return _set_first_key(ret, 'name', name)

    @property
    def size_in_bytes(self):
        if False:
            return 10
        return self.value_type.size_in_bytes * self.length

    @property
    def subtype(self):
        if False:
            print('Hello World!')
        '\n        Alias for API compatibility with codegen\n        '
        return self.value_type

    def get_subscripted_type(self, node):
        if False:
            return 10
        return self.value_type

    def compare_type(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(self, type(other)):
            return False
        if self.length != other.length:
            return False
        return self.value_type.compare_type(other.value_type)

    @classmethod
    def from_annotation(cls, node: vy_ast.Subscript) -> 'SArrayT':
        if False:
            return 10
        if not isinstance(node, vy_ast.Subscript) or not isinstance(node.slice, vy_ast.Index):
            raise StructureException('Arrays must be defined with base type and length, e.g. bool[5]', node)
        value_type = type_from_annotation(node.value)
        if not value_type._as_array:
            raise StructureException(f'arrays of {value_type} are not allowed!')
        length = get_index_value(node.slice)
        return cls(value_type, length)

class DArrayT(_SequenceT):
    """
    Dynamic array type
    """
    _valid_literal = (vy_ast.List,)
    _as_array = True
    _id = 'DynArray'

    def __init__(self, value_type: VyperType, length: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(value_type, length)
        from vyper.semantics.types.function import MemberFunctionT
        self.add_member('append', MemberFunctionT(self, 'append', [self.value_type], None, True))
        self.add_member('pop', MemberFunctionT(self, 'pop', [], self.value_type, True))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'DynArray[{self.value_type}, {self.length}]'

    @property
    def subtype(self):
        if False:
            while True:
                i = 10
        '\n        Alias for backwards compatibility.\n        '
        return self.value_type

    @property
    def count(self):
        if False:
            i = 10
            return i + 15
        '\n        Alias for backwards compatibility.\n        '
        return self.length

    @property
    def abi_type(self) -> ABIType:
        if False:
            for i in range(10):
                print('nop')
        return ABI_DynamicArray(self.value_type.abi_type, self.length)

    def to_abi_arg(self, name: str='') -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        ret = self.value_type.to_abi_arg()
        ret['type'] += '[]'
        return _set_first_key(ret, 'name', name)

    @property
    def size_in_bytes(self):
        if False:
            return 10
        return 32 + self.value_type.size_in_bytes * self.length

    def compare_type(self, other):
        if False:
            return 10
        if not isinstance(self, type(other)):
            return False
        if self.length < other.length:
            return False
        return self.value_type.compare_type(other.value_type)

    @classmethod
    def from_annotation(cls, node: vy_ast.Subscript) -> 'DArrayT':
        if False:
            print('Hello World!')
        if not isinstance(node, vy_ast.Subscript) or not isinstance(node.slice, vy_ast.Index) or (not isinstance(node.slice.value, vy_ast.Tuple)) or (not isinstance(node.slice.value.elements[1], vy_ast.Int)) or (len(node.slice.value.elements) != 2):
            raise StructureException('DynArray must be defined with base type and max length, e.g. DynArray[bool, 5]', node)
        value_type = type_from_annotation(node.slice.value.elements[0])
        if not value_type._as_darray:
            raise StructureException(f'Arrays of {value_type} are not allowed', node)
        max_length = node.slice.value.elements[1].value
        return cls(value_type, max_length)

class TupleT(VyperType):
    """
    Tuple type definition.

    This class is used to represent multiple return values from functions.
    """
    _equality_attrs = ('members',)

    def __init__(self, member_types: Tuple[VyperType, ...]) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.member_types = member_types
        self.key_type = UINT256_T

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '(' + ', '.join((repr(t) for t in self.member_types)) + ')'

    @property
    def length(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.member_types)

    def tuple_members(self):
        if False:
            print('Hello World!')
        return [v for (_k, v) in self.tuple_items()]

    def tuple_keys(self):
        if False:
            while True:
                i = 10
        return [k for (k, _v) in self.tuple_items()]

    def tuple_items(self):
        if False:
            return 10
        return list(enumerate(self.member_types))

    @classmethod
    def from_annotation(cls, node: vy_ast.Tuple) -> VyperType:
        if False:
            i = 10
            return i + 15
        values = node.elements
        types = tuple((type_from_annotation(v) for v in values))
        return cls(types)

    @property
    def abi_type(self) -> ABIType:
        if False:
            return 10
        return ABI_Tuple([t.abi_type for t in self.member_types])

    def to_abi_arg(self, name: str='') -> dict:
        if False:
            i = 10
            return i + 15
        components = [t.to_abi_arg() for t in self.member_types]
        return {'name': name, 'type': 'tuple', 'components': components}

    @property
    def size_in_bytes(self):
        if False:
            print('Hello World!')
        return sum((i.size_in_bytes for i in self.member_types))

    def validate_index_type(self, node):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(node, vy_ast.Int):
            raise InvalidType('Tuple indexes must be literals', node)
        if node.value < 0:
            raise ArrayIndexException('Vyper does not support negative indexing', node)
        if node.value >= self.length:
            raise ArrayIndexException('Index out of range', node)

    def get_subscripted_type(self, node):
        if False:
            i = 10
            return i + 15
        return self.member_types[node.value]

    def compare_type(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(self, type(other)):
            return False
        if self.length != other.length:
            return False
        return all((a.compare_type(b) for (a, b) in zip(self.member_types, other.member_types)))