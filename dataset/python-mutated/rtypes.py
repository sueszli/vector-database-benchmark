"""Types used in the intermediate representation.

These are runtime types (RTypes), as opposed to mypy Type objects.
The latter are only used during type checking and not directly used at
runtime.  Runtime types are derived from mypy types, but there's no
simple one-to-one correspondence. (Here 'runtime' means 'runtime
checked'.)

The generated IR ensures some runtime type safety properties based on
RTypes. Compiled code can assume that the runtime value matches the
static RType of a value. If the RType of a register is 'builtins.str'
(str_rprimitive), for example, the generated IR will ensure that the
register will have a 'str' object.

RTypes are simpler and less expressive than mypy (or PEP 484)
types. For example, all mypy types of form 'list[T]' (for arbitrary T)
are erased to the single RType 'builtins.list' (list_rprimitive).

mypyc.irbuild.mapper.Mapper.type_to_rtype converts mypy Types to mypyc
RTypes.
"""
from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Final, Generic, TypeVar
from typing_extensions import TypeGuard
from mypyc.common import IS_32_BIT_PLATFORM, PLATFORM_SIZE, JsonDict, short_name
from mypyc.namegen import NameGenerator
if TYPE_CHECKING:
    from mypyc.ir.class_ir import ClassIR
    from mypyc.ir.ops import DeserMaps
T = TypeVar('T')

class RType:
    """Abstract base class for runtime types (erased, only concrete; no generics)."""
    name: str
    is_unboxed = False
    c_undefined: str
    is_refcounted = True
    _ctype: str
    error_overlap = False

    @abstractmethod
    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def short_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return short_name(self.name)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return short_name(self.name)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '<%s>' % self.__class__.__name__

    def serialize(self) -> JsonDict | str:
        if False:
            return 10
        raise NotImplementedError(f'Cannot serialize {self.__class__.__name__} instance')

def deserialize_type(data: JsonDict | str, ctx: DeserMaps) -> RType:
    if False:
        i = 10
        return i + 15
    'Deserialize a JSON-serialized RType.\n\n    Arguments:\n        data: The decoded JSON of the serialized type\n        ctx: The deserialization maps to use\n    '
    if isinstance(data, str):
        if data in ctx.classes:
            return RInstance(ctx.classes[data])
        elif data in RPrimitive.primitive_map:
            return RPrimitive.primitive_map[data]
        elif data == 'void':
            return RVoid()
        else:
            assert False, f"Can't find class {data}"
    elif data['.class'] == 'RTuple':
        return RTuple.deserialize(data, ctx)
    elif data['.class'] == 'RUnion':
        return RUnion.deserialize(data, ctx)
    raise NotImplementedError('unexpected .class {}'.format(data['.class']))

class RTypeVisitor(Generic[T]):
    """Generic visitor over RTypes (uses the visitor design pattern)."""

    @abstractmethod
    def visit_rprimitive(self, typ: RPrimitive) -> T:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def visit_rinstance(self, typ: RInstance) -> T:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def visit_runion(self, typ: RUnion) -> T:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def visit_rtuple(self, typ: RTuple) -> T:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @abstractmethod
    def visit_rstruct(self, typ: RStruct) -> T:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def visit_rarray(self, typ: RArray) -> T:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def visit_rvoid(self, typ: RVoid) -> T:
        if False:
            print('Hello World!')
        raise NotImplementedError

class RVoid(RType):
    """The void type (no value).

    This is a singleton -- use void_rtype (below) to refer to this instead of
    constructing a new instance.
    """
    is_unboxed = False
    name = 'void'
    ctype = 'void'

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            return 10
        return visitor.visit_rvoid(self)

    def serialize(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'void'

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return isinstance(other, RVoid)

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(RVoid)
void_rtype: Final = RVoid()

class RPrimitive(RType):
    """Primitive type such as 'object' or 'int'.

    These often have custom ops associated with them. The 'object'
    primitive type can be used to hold arbitrary Python objects.

    Different primitive types have different representations, and
    primitives may be unboxed or boxed. Primitive types don't need to
    directly correspond to Python types, but most do.

    NOTE: All supported primitive types are defined below
    (e.g. object_rprimitive).
    """
    primitive_map: ClassVar[dict[str, RPrimitive]] = {}

    def __init__(self, name: str, *, is_unboxed: bool, is_refcounted: bool, is_native_int: bool=False, is_signed: bool=False, ctype: str='PyObject *', size: int=PLATFORM_SIZE, error_overlap: bool=False) -> None:
        if False:
            return 10
        RPrimitive.primitive_map[name] = self
        self.name = name
        self.is_unboxed = is_unboxed
        self.is_refcounted = is_refcounted
        self.is_native_int = is_native_int
        self.is_signed = is_signed
        self._ctype = ctype
        self.size = size
        self.error_overlap = error_overlap
        if ctype == 'CPyTagged':
            self.c_undefined = 'CPY_INT_TAG'
        elif ctype in ('int16_t', 'int32_t', 'int64_t'):
            self.c_undefined = '-113'
        elif ctype == 'CPyPtr':
            self.c_undefined = '0'
        elif ctype == 'PyObject *':
            self.c_undefined = 'NULL'
        elif ctype == 'char':
            self.c_undefined = '2'
        elif ctype in ('PyObject **', 'void *'):
            self.c_undefined = 'NULL'
        elif ctype == 'double':
            self.c_undefined = '-113.0'
        elif ctype in ('uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'):
            self.c_undefined = '239'
        else:
            assert False, 'Unrecognized ctype: %r' % ctype

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            return 10
        return visitor.visit_rprimitive(self)

    def serialize(self) -> str:
        if False:
            while True:
                i = 10
        return self.name

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '<RPrimitive %s>' % self.name

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(other, RPrimitive) and other.name == self.name

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self.name)
object_rprimitive: Final = RPrimitive('builtins.object', is_unboxed=False, is_refcounted=True)
object_pointer_rprimitive: Final = RPrimitive('object_ptr', is_unboxed=False, is_refcounted=False, ctype='PyObject **')
int_rprimitive: Final = RPrimitive('builtins.int', is_unboxed=True, is_refcounted=True, ctype='CPyTagged')
short_int_rprimitive: Final = RPrimitive('short_int', is_unboxed=True, is_refcounted=False, ctype='CPyTagged')
int16_rprimitive: Final = RPrimitive('i16', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=True, ctype='int16_t', size=2, error_overlap=True)
int32_rprimitive: Final = RPrimitive('i32', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=True, ctype='int32_t', size=4, error_overlap=True)
int64_rprimitive: Final = RPrimitive('i64', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=True, ctype='int64_t', size=8, error_overlap=True)
uint8_rprimitive: Final = RPrimitive('u8', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=False, ctype='uint8_t', size=1, error_overlap=True)
u16_rprimitive: Final = RPrimitive('u16', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=False, ctype='uint16_t', size=2, error_overlap=True)
uint32_rprimitive: Final = RPrimitive('u32', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=False, ctype='uint32_t', size=4, error_overlap=True)
uint64_rprimitive: Final = RPrimitive('u64', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=False, ctype='uint64_t', size=8, error_overlap=True)
c_int_rprimitive = int32_rprimitive
if IS_32_BIT_PLATFORM:
    c_size_t_rprimitive = uint32_rprimitive
    c_pyssize_t_rprimitive = RPrimitive('native_int', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=True, ctype='int32_t', size=4)
else:
    c_size_t_rprimitive = uint64_rprimitive
    c_pyssize_t_rprimitive = RPrimitive('native_int', is_unboxed=True, is_refcounted=False, is_native_int=True, is_signed=True, ctype='int64_t', size=8)
pointer_rprimitive: Final = RPrimitive('ptr', is_unboxed=True, is_refcounted=False, ctype='CPyPtr')
c_pointer_rprimitive: Final = RPrimitive('c_ptr', is_unboxed=False, is_refcounted=False, ctype='void *')
bitmap_rprimitive: Final = uint32_rprimitive
float_rprimitive: Final = RPrimitive('builtins.float', is_unboxed=True, is_refcounted=False, ctype='double', size=8, error_overlap=True)
bool_rprimitive: Final = RPrimitive('builtins.bool', is_unboxed=True, is_refcounted=False, ctype='char', size=1)
bit_rprimitive: Final = RPrimitive('bit', is_unboxed=True, is_refcounted=False, ctype='char', size=1)
none_rprimitive: Final = RPrimitive('builtins.None', is_unboxed=True, is_refcounted=False, ctype='char', size=1)
list_rprimitive: Final = RPrimitive('builtins.list', is_unboxed=False, is_refcounted=True)
dict_rprimitive: Final = RPrimitive('builtins.dict', is_unboxed=False, is_refcounted=True)
set_rprimitive: Final = RPrimitive('builtins.set', is_unboxed=False, is_refcounted=True)
str_rprimitive: Final = RPrimitive('builtins.str', is_unboxed=False, is_refcounted=True)
bytes_rprimitive: Final = RPrimitive('builtins.bytes', is_unboxed=False, is_refcounted=True)
tuple_rprimitive: Final = RPrimitive('builtins.tuple', is_unboxed=False, is_refcounted=True)
range_rprimitive: Final = RPrimitive('builtins.range', is_unboxed=False, is_refcounted=True)

def is_tagged(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return rtype is int_rprimitive or rtype is short_int_rprimitive

def is_int_rprimitive(rtype: RType) -> bool:
    if False:
        i = 10
        return i + 15
    return rtype is int_rprimitive

def is_short_int_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return rtype is short_int_rprimitive

def is_int16_rprimitive(rtype: RType) -> TypeGuard[RPrimitive]:
    if False:
        print('Hello World!')
    return rtype is int16_rprimitive

def is_int32_rprimitive(rtype: RType) -> TypeGuard[RPrimitive]:
    if False:
        print('Hello World!')
    return rtype is int32_rprimitive or (rtype is c_pyssize_t_rprimitive and rtype._ctype == 'int32_t')

def is_int64_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return rtype is int64_rprimitive or (rtype is c_pyssize_t_rprimitive and rtype._ctype == 'int64_t')

def is_fixed_width_rtype(rtype: RType) -> TypeGuard[RPrimitive]:
    if False:
        i = 10
        return i + 15
    return is_int64_rprimitive(rtype) or is_int32_rprimitive(rtype) or is_int16_rprimitive(rtype) or is_uint8_rprimitive(rtype)

def is_uint8_rprimitive(rtype: RType) -> TypeGuard[RPrimitive]:
    if False:
        print('Hello World!')
    return rtype is uint8_rprimitive

def is_uint32_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return rtype is uint32_rprimitive

def is_uint64_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return rtype is uint64_rprimitive

def is_c_py_ssize_t_rprimitive(rtype: RType) -> bool:
    if False:
        print('Hello World!')
    return rtype is c_pyssize_t_rprimitive

def is_pointer_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return rtype is pointer_rprimitive

def is_float_rprimitive(rtype: RType) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.float'

def is_bool_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.bool'

def is_bit_rprimitive(rtype: RType) -> bool:
    if False:
        return 10
    return isinstance(rtype, RPrimitive) and rtype.name == 'bit'

def is_object_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.object'

def is_none_rprimitive(rtype: RType) -> bool:
    if False:
        return 10
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.None'

def is_list_rprimitive(rtype: RType) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.list'

def is_dict_rprimitive(rtype: RType) -> bool:
    if False:
        print('Hello World!')
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.dict'

def is_set_rprimitive(rtype: RType) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.set'

def is_str_rprimitive(rtype: RType) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.str'

def is_bytes_rprimitive(rtype: RType) -> bool:
    if False:
        print('Hello World!')
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.bytes'

def is_tuple_rprimitive(rtype: RType) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.tuple'

def is_range_rprimitive(rtype: RType) -> bool:
    if False:
        return 10
    return isinstance(rtype, RPrimitive) and rtype.name == 'builtins.range'

def is_sequence_rprimitive(rtype: RType) -> bool:
    if False:
        return 10
    return isinstance(rtype, RPrimitive) and (is_list_rprimitive(rtype) or is_tuple_rprimitive(rtype) or is_str_rprimitive(rtype))

class TupleNameVisitor(RTypeVisitor[str]):
    """Produce a tuple name based on the concrete representations of types."""

    def visit_rinstance(self, t: RInstance) -> str:
        if False:
            while True:
                i = 10
        return 'O'

    def visit_runion(self, t: RUnion) -> str:
        if False:
            while True:
                i = 10
        return 'O'

    def visit_rprimitive(self, t: RPrimitive) -> str:
        if False:
            i = 10
            return i + 15
        if t._ctype == 'CPyTagged':
            return 'I'
        elif t._ctype == 'char':
            return 'C'
        elif t._ctype == 'int64_t':
            return '8'
        elif t._ctype == 'int32_t':
            return '4'
        elif t._ctype == 'int16_t':
            return '2'
        elif t._ctype == 'uint8_t':
            return 'U1'
        elif t._ctype == 'double':
            return 'F'
        assert not t.is_unboxed, f'{t} unexpected unboxed type'
        return 'O'

    def visit_rtuple(self, t: RTuple) -> str:
        if False:
            return 10
        parts = [elem.accept(self) for elem in t.types]
        return 'T{}{}'.format(len(parts), ''.join(parts))

    def visit_rstruct(self, t: RStruct) -> str:
        if False:
            return 10
        assert False, 'RStruct not supported in tuple'

    def visit_rarray(self, t: RArray) -> str:
        if False:
            while True:
                i = 10
        assert False, 'RArray not supported in tuple'

    def visit_rvoid(self, t: RVoid) -> str:
        if False:
            i = 10
            return i + 15
        assert False, 'rvoid in tuple?'

class RTuple(RType):
    """Fixed-length unboxed tuple (represented as a C struct).

    These are used to represent mypy TupleType values (fixed-length
    Python tuples). Since this is unboxed, the identity of a tuple
    object is not preserved within compiled code. If the identity of a
    tuple is important, or there is a need to have multiple references
    to a single tuple object, a variable-length tuple should be used
    (tuple_rprimitive or Tuple[T, ...]  with explicit '...'), as they
    are boxed.

    These aren't immutable. However, user code won't be able to mutate
    individual tuple items.
    """
    is_unboxed = True

    def __init__(self, types: list[RType]) -> None:
        if False:
            print('Hello World!')
        self.name = 'tuple'
        self.types = tuple(types)
        self.is_refcounted = any((t.is_refcounted for t in self.types))
        self.unique_id = self.accept(TupleNameVisitor())
        self.struct_name = f'tuple_{self.unique_id}'
        self._ctype = f'{self.struct_name}'
        self.error_overlap = all((t.error_overlap for t in self.types)) and bool(self.types)

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            while True:
                i = 10
        return visitor.visit_rtuple(self)

    def __str__(self) -> str:
        if False:
            return 10
        return 'tuple[%s]' % ', '.join((str(typ) for typ in self.types))

    def __repr__(self) -> str:
        if False:
            return 10
        return '<RTuple %s>' % ', '.join((repr(typ) for typ in self.types))

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(other, RTuple) and self.types == other.types

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash((self.name, self.types))

    def serialize(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        types = [x.serialize() for x in self.types]
        return {'.class': 'RTuple', 'types': types}

    @classmethod
    def deserialize(cls, data: JsonDict, ctx: DeserMaps) -> RTuple:
        if False:
            for i in range(10):
                print('nop')
        types = [deserialize_type(t, ctx) for t in data['types']]
        return RTuple(types)
exc_rtuple = RTuple([object_rprimitive, object_rprimitive, object_rprimitive])
dict_next_rtuple_pair = RTuple([bool_rprimitive, short_int_rprimitive, object_rprimitive, object_rprimitive])
dict_next_rtuple_single = RTuple([bool_rprimitive, short_int_rprimitive, object_rprimitive])

def compute_rtype_alignment(typ: RType) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Compute alignment of a given type based on platform alignment rule'
    platform_alignment = PLATFORM_SIZE
    if isinstance(typ, RPrimitive):
        return typ.size
    elif isinstance(typ, RInstance):
        return platform_alignment
    elif isinstance(typ, RUnion):
        return platform_alignment
    elif isinstance(typ, RArray):
        return compute_rtype_alignment(typ.item_type)
    else:
        if isinstance(typ, RTuple):
            items = list(typ.types)
        elif isinstance(typ, RStruct):
            items = typ.types
        else:
            assert False, 'invalid rtype for computing alignment'
        max_alignment = max((compute_rtype_alignment(item) for item in items))
        return max_alignment

def compute_rtype_size(typ: RType) -> int:
    if False:
        print('Hello World!')
    'Compute unaligned size of rtype'
    if isinstance(typ, RPrimitive):
        return typ.size
    elif isinstance(typ, RTuple):
        return compute_aligned_offsets_and_size(list(typ.types))[1]
    elif isinstance(typ, RUnion):
        return PLATFORM_SIZE
    elif isinstance(typ, RStruct):
        return compute_aligned_offsets_and_size(typ.types)[1]
    elif isinstance(typ, RInstance):
        return PLATFORM_SIZE
    elif isinstance(typ, RArray):
        alignment = compute_rtype_alignment(typ)
        aligned_size = compute_rtype_size(typ.item_type) + (alignment - 1) & ~(alignment - 1)
        return aligned_size * typ.length
    else:
        assert False, 'invalid rtype for computing size'

def compute_aligned_offsets_and_size(types: list[RType]) -> tuple[list[int], int]:
    if False:
        for i in range(10):
            print('nop')
    'Compute offsets and total size of a list of types after alignment\n\n    Note that the types argument are types of values that are stored\n    sequentially with platform default alignment.\n    '
    unaligned_sizes = [compute_rtype_size(typ) for typ in types]
    alignments = [compute_rtype_alignment(typ) for typ in types]
    current_offset = 0
    offsets = []
    final_size = 0
    for i in range(len(unaligned_sizes)):
        offsets.append(current_offset)
        if i + 1 < len(unaligned_sizes):
            cur_size = unaligned_sizes[i]
            current_offset += cur_size
            next_alignment = alignments[i + 1]
            current_offset = current_offset + (next_alignment - 1) & -next_alignment
        else:
            struct_alignment = max(alignments)
            final_size = current_offset + unaligned_sizes[i]
            final_size = final_size + (struct_alignment - 1) & -struct_alignment
    return (offsets, final_size)

class RStruct(RType):
    """C struct type"""

    def __init__(self, name: str, names: list[str], types: list[RType]) -> None:
        if False:
            while True:
                i = 10
        self.name = name
        self.names = names
        self.types = types
        if len(self.names) < len(self.types):
            for i in range(len(self.types) - len(self.names)):
                self.names.append('_item' + str(i))
        (self.offsets, self.size) = compute_aligned_offsets_and_size(types)
        self._ctype = name

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            i = 10
            return i + 15
        return visitor.visit_rstruct(self)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return '{}{{{}}}'.format(self.name, ', '.join((name + ':' + str(typ) for (name, typ) in zip(self.names, self.types))))

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<RStruct {}{{{}}}>'.format(self.name, ', '.join((name + ':' + repr(typ) for (name, typ) in zip(self.names, self.types))))

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, RStruct) and self.name == other.name and (self.names == other.names) and (self.types == other.types)

    def __hash__(self) -> int:
        if False:
            return 10
        return hash((self.name, tuple(self.names), tuple(self.types)))

    def serialize(self) -> JsonDict:
        if False:
            return 10
        assert False

    @classmethod
    def deserialize(cls, data: JsonDict, ctx: DeserMaps) -> RStruct:
        if False:
            print('Hello World!')
        assert False

class RInstance(RType):
    """Instance of user-defined class (compiled to C extension class).

    The runtime representation is 'PyObject *', and these are always
    boxed and thus reference-counted.

    These support fast method calls and fast attribute access using
    vtables, and they usually use a dict-free, struct-based
    representation of attributes. Method calls and attribute access
    can skip the vtable if we know that there is no overriding.

    These are also sometimes called 'native' types, since these have
    the most efficient representation and ops (along with certain
    RPrimitive types and RTuple).
    """
    is_unboxed = False

    def __init__(self, class_ir: ClassIR) -> None:
        if False:
            return 10
        self.name = class_ir.fullname
        self.class_ir = class_ir
        self._ctype = 'PyObject *'

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            print('Hello World!')
        return visitor.visit_rinstance(self)

    def struct_name(self, names: NameGenerator) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.class_ir.struct_name(names)

    def getter_index(self, name: str) -> int:
        if False:
            i = 10
            return i + 15
        return self.class_ir.vtable_entry(name)

    def setter_index(self, name: str) -> int:
        if False:
            return 10
        return self.getter_index(name) + 1

    def method_index(self, name: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.class_ir.vtable_entry(name)

    def attr_type(self, name: str) -> RType:
        if False:
            i = 10
            return i + 15
        return self.class_ir.attr_type(name)

    def __repr__(self) -> str:
        if False:
            return 10
        return '<RInstance %s>' % self.name

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, RInstance) and other.name == self.name

    def __hash__(self) -> int:
        if False:
            return 10
        return hash(self.name)

    def serialize(self) -> str:
        if False:
            while True:
                i = 10
        return self.name

class RUnion(RType):
    """union[x, ..., y]"""
    is_unboxed = False

    def __init__(self, items: list[RType]) -> None:
        if False:
            while True:
                i = 10
        self.name = 'union'
        self.items = items
        self.items_set = frozenset(items)
        self._ctype = 'PyObject *'

    @staticmethod
    def make_simplified_union(items: list[RType]) -> RType:
        if False:
            return 10
        'Return a normalized union that covers the given items.\n\n        Flatten nested unions and remove duplicate items.\n\n        Overlapping items are *not* simplified. For example,\n        [object, str] will not be simplified.\n        '
        items = flatten_nested_unions(items)
        assert items
        unique_items = dict.fromkeys(items)
        if len(unique_items) > 1:
            return RUnion(list(unique_items))
        else:
            return next(iter(unique_items))

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            for i in range(10):
                print('nop')
        return visitor.visit_runion(self)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<RUnion %s>' % ', '.join((str(item) for item in self.items))

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'union[%s]' % ', '.join((str(item) for item in self.items))

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, RUnion) and self.items_set == other.items_set

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash(('union', self.items_set))

    def serialize(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        types = [x.serialize() for x in self.items]
        return {'.class': 'RUnion', 'types': types}

    @classmethod
    def deserialize(cls, data: JsonDict, ctx: DeserMaps) -> RUnion:
        if False:
            for i in range(10):
                print('nop')
        types = [deserialize_type(t, ctx) for t in data['types']]
        return RUnion(types)

def flatten_nested_unions(types: list[RType]) -> list[RType]:
    if False:
        return 10
    if not any((isinstance(t, RUnion) for t in types)):
        return types
    flat_items: list[RType] = []
    for t in types:
        if isinstance(t, RUnion):
            flat_items.extend(flatten_nested_unions(t.items))
        else:
            flat_items.append(t)
    return flat_items

def optional_value_type(rtype: RType) -> RType | None:
    if False:
        print('Hello World!')
    'If rtype is the union of none_rprimitive and another type X, return X.\n\n    Otherwise return None.\n    '
    if isinstance(rtype, RUnion) and len(rtype.items) == 2:
        if rtype.items[0] == none_rprimitive:
            return rtype.items[1]
        elif rtype.items[1] == none_rprimitive:
            return rtype.items[0]
    return None

def is_optional_type(rtype: RType) -> bool:
    if False:
        return 10
    'Is rtype an optional type with exactly two union items?'
    return optional_value_type(rtype) is not None

class RArray(RType):
    """Fixed-length C array type (for example, int[5]).

    Note that the implementation is a bit limited, and these can basically
    be only used for local variables that are initialized in one location.
    """

    def __init__(self, item_type: RType, length: int) -> None:
        if False:
            print('Hello World!')
        self.item_type = item_type
        self.length = length
        self.is_refcounted = False

    def accept(self, visitor: RTypeVisitor[T]) -> T:
        if False:
            print('Hello World!')
        return visitor.visit_rarray(self)

    def __str__(self) -> str:
        if False:
            return 10
        return f'{self.item_type}[{self.length}]'

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<RArray {self.item_type!r}[{self.length}]>'

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(other, RArray) and self.item_type == other.item_type and (self.length == other.length)

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash((self.item_type, self.length))

    def serialize(self) -> JsonDict:
        if False:
            print('Hello World!')
        assert False

    @classmethod
    def deserialize(cls, data: JsonDict, ctx: DeserMaps) -> RArray:
        if False:
            while True:
                i = 10
        assert False
PyObject = RStruct(name='PyObject', names=['ob_refcnt', 'ob_type'], types=[c_pyssize_t_rprimitive, pointer_rprimitive])
PyVarObject = RStruct(name='PyVarObject', names=['ob_base', 'ob_size'], types=[PyObject, c_pyssize_t_rprimitive])
setentry = RStruct(name='setentry', names=['key', 'hash'], types=[pointer_rprimitive, c_pyssize_t_rprimitive])
smalltable = RStruct(name='smalltable', names=[], types=[setentry] * 8)
PySetObject = RStruct(name='PySetObject', names=['ob_base', 'fill', 'used', 'mask', 'table', 'hash', 'finger', 'smalltable', 'weakreflist'], types=[PyObject, c_pyssize_t_rprimitive, c_pyssize_t_rprimitive, c_pyssize_t_rprimitive, pointer_rprimitive, c_pyssize_t_rprimitive, c_pyssize_t_rprimitive, smalltable, pointer_rprimitive])
PyListObject = RStruct(name='PyListObject', names=['ob_base', 'ob_item', 'allocated'], types=[PyVarObject, pointer_rprimitive, c_pyssize_t_rprimitive])

def check_native_int_range(rtype: RPrimitive, n: int) -> bool:
    if False:
        print('Hello World!')
    'Is n within the range of a native, fixed-width int type?\n\n    Assume the type is a fixed-width int type.\n    '
    if not rtype.is_signed:
        return 0 <= n < 1 << 8 * rtype.size
    else:
        limit = 1 << rtype.size * 8 - 1
        return -limit <= n < limit