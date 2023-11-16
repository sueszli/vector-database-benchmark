"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments

class SpecialArgName(Enum):
    possibly_redundant_memory_format = auto()
ArgName = Union[str, SpecialArgName]

@dataclass(frozen=True)
class BaseCppType:
    ns: Optional[str]
    name: str

    def __str__(self) -> str:
        if False:
            return 10
        if self.ns is None or self.ns == '':
            return self.name
        return f'{self.ns}::{self.name}'
byteT = BaseCppType('', 'uint8_t')
charT = BaseCppType('', 'int8_t')
shortT = BaseCppType('', 'int16_t')
int32T = BaseCppType('', 'int32_t')
longT = BaseCppType('', 'int64_t')
doubleT = BaseCppType('', 'double')
floatT = BaseCppType('', 'float')
boolT = BaseCppType('', 'bool')
voidT = BaseCppType('', 'void')

class CType(ABC):

    @abstractmethod
    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def cpp_type_registration_declarations(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def remove_const_ref(self) -> 'CType':
        if False:
            print('Hello World!')
        return self

@dataclass(frozen=True)
class BaseCType(CType):
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        return str(self.type)

    def cpp_type_registration_declarations(self) -> str:
        if False:
            return 10
        return str(self.type).replace('at::', '')

    def remove_const_ref(self) -> 'CType':
        if False:
            print('Hello World!')
        return self

@dataclass(frozen=True)
class ConstRefCType(CType):
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'const {self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        if False:
            print('Hello World!')
        return f'const {self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        if False:
            i = 10
            return i + 15
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class VectorCType(CType):
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'::std::vector<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'::std::vector<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        if False:
            print('Hello World!')
        return VectorCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayCType(CType):
    elem: 'CType'
    size: int

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'::std::array<{self.elem.cpp_type()},{self.size}>'

    def cpp_type_registration_declarations(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>'

    def remove_const_ref(self) -> 'CType':
        if False:
            i = 10
            return i + 15
        return ArrayCType(self.elem.remove_const_ref(), self.size)

@dataclass(frozen=True)
class TupleCType(CType):
    elems: List['CType']

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            while True:
                i = 10
        return f"::std::tuple<{','.join([e.cpp_type() for e in self.elems])}>"

    def cpp_type_registration_declarations(self) -> str:
        if False:
            print('Hello World!')
        return f"::std::tuple<{','.join([e.cpp_type_registration_declarations() for e in self.elems])}>"

    def remove_const_ref(self) -> 'CType':
        if False:
            i = 10
            return i + 15
        return TupleCType([e.remove_const_ref() for e in self.elems])

@dataclass(frozen=True)
class MutRefCType(CType):
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            while True:
                i = 10
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'{self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        if False:
            return 10
        return f'{self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        if False:
            return 10
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if False:
            return 10
        return self.type.cpp_type(strip_ref=strip_ref)

    def cpp_type_registration_declarations(self) -> str:
        if False:
            print('Hello World!')
        return self.type.cpp_type_registration_declarations()

    def remove_const_ref(self) -> 'NamedCType':
        if False:
            return 10
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> 'NamedCType':
        if False:
            return 10
        return NamedCType(name, self.type)

@dataclass(frozen=True)
class Binding:
    name: str
    nctype: NamedCType
    argument: Union[Argument, TensorOptionsArguments, SelfArgument]
    default: Optional[str] = None

    def rename(self, name: str) -> 'Binding':
        if False:
            return 10
        return Binding(name=name, nctype=self.nctype, argument=self.argument, default=self.default)

    @property
    def type(self) -> str:
        if False:
            return 10
        return self.nctype.cpp_type()

    def no_default(self) -> 'Binding':
        if False:
            return 10
        return Binding(name=self.name, nctype=self.nctype, default=None, argument=self.argument)

    def decl(self, *, func_ptr_cast: bool=False) -> str:
        if False:
            return 10
        mb_default = ''
        if self.default is not None:
            mb_default = f'={self.default}'
        if func_ptr_cast:
            return f'{self.type}'
        else:
            return f'{self.type} {self.name}{mb_default}'

    def decl_registration_declarations(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        type_s = self.nctype.cpp_type_registration_declarations()
        mb_default = ''
        if self.default is not None:
            mb_default = f'={self.default}'
        return f'{type_s} {self.name}{mb_default}'

    def defn(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.type} {self.name}'

    def with_name(self, name: str) -> 'Binding':
        if False:
            while True:
                i = 10
        return Binding(name=name, nctype=self.nctype, argument=self.argument, default=self.default)

@dataclass(frozen=True)
class Expr:
    expr: str
    type: NamedCType