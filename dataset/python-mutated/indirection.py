from __future__ import annotations
from typing import Iterable, Set
import mypy.types as types
from mypy.types import TypeVisitor
from mypy.util import split_module_names

def extract_module_names(type_name: str | None) -> list[str]:
    if False:
        i = 10
        return i + 15
    'Returns the module names of a fully qualified type name.'
    if type_name is not None:
        possible_module_names = split_module_names(type_name)
        return possible_module_names[1:]
    else:
        return []

class TypeIndirectionVisitor(TypeVisitor[Set[str]]):
    """Returns all module references within a particular type."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.cache: dict[types.Type, set[str]] = {}
        self.seen_aliases: set[types.TypeAliasType] = set()

    def find_modules(self, typs: Iterable[types.Type]) -> set[str]:
        if False:
            return 10
        self.seen_aliases.clear()
        return self._visit(typs)

    def _visit(self, typ_or_typs: types.Type | Iterable[types.Type]) -> set[str]:
        if False:
            while True:
                i = 10
        typs = [typ_or_typs] if isinstance(typ_or_typs, types.Type) else typ_or_typs
        output: set[str] = set()
        for typ in typs:
            if isinstance(typ, types.TypeAliasType):
                if typ in self.seen_aliases:
                    continue
                self.seen_aliases.add(typ)
            if typ in self.cache:
                modules = self.cache[typ]
            else:
                modules = typ.accept(self)
                self.cache[typ] = set(modules)
            output.update(modules)
        return output

    def visit_unbound_type(self, t: types.UnboundType) -> set[str]:
        if False:
            i = 10
            return i + 15
        return self._visit(t.args)

    def visit_any(self, t: types.AnyType) -> set[str]:
        if False:
            return 10
        return set()

    def visit_none_type(self, t: types.NoneType) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        return set()

    def visit_uninhabited_type(self, t: types.UninhabitedType) -> set[str]:
        if False:
            while True:
                i = 10
        return set()

    def visit_erased_type(self, t: types.ErasedType) -> set[str]:
        if False:
            i = 10
            return i + 15
        return set()

    def visit_deleted_type(self, t: types.DeletedType) -> set[str]:
        if False:
            while True:
                i = 10
        return set()

    def visit_type_var(self, t: types.TypeVarType) -> set[str]:
        if False:
            return 10
        return self._visit(t.values) | self._visit(t.upper_bound) | self._visit(t.default)

    def visit_param_spec(self, t: types.ParamSpecType) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._visit(t.upper_bound) | self._visit(t.default)

    def visit_type_var_tuple(self, t: types.TypeVarTupleType) -> set[str]:
        if False:
            while True:
                i = 10
        return self._visit(t.upper_bound) | self._visit(t.default)

    def visit_unpack_type(self, t: types.UnpackType) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        return t.type.accept(self)

    def visit_parameters(self, t: types.Parameters) -> set[str]:
        if False:
            while True:
                i = 10
        return self._visit(t.arg_types)

    def visit_instance(self, t: types.Instance) -> set[str]:
        if False:
            i = 10
            return i + 15
        out = self._visit(t.args)
        if t.type:
            for s in t.type.mro:
                out.update(split_module_names(s.module_name))
            if t.type.metaclass_type is not None:
                out.update(split_module_names(t.type.metaclass_type.type.module_name))
        return out

    def visit_callable_type(self, t: types.CallableType) -> set[str]:
        if False:
            return 10
        out = self._visit(t.arg_types) | self._visit(t.ret_type)
        if t.definition is not None:
            out.update(extract_module_names(t.definition.fullname))
        return out

    def visit_overloaded(self, t: types.Overloaded) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._visit(t.items) | self._visit(t.fallback)

    def visit_tuple_type(self, t: types.TupleType) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._visit(t.items) | self._visit(t.partial_fallback)

    def visit_typeddict_type(self, t: types.TypedDictType) -> set[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._visit(t.items.values()) | self._visit(t.fallback)

    def visit_literal_type(self, t: types.LiteralType) -> set[str]:
        if False:
            return 10
        return self._visit(t.fallback)

    def visit_union_type(self, t: types.UnionType) -> set[str]:
        if False:
            i = 10
            return i + 15
        return self._visit(t.items)

    def visit_partial_type(self, t: types.PartialType) -> set[str]:
        if False:
            print('Hello World!')
        return set()

    def visit_type_type(self, t: types.TypeType) -> set[str]:
        if False:
            while True:
                i = 10
        return self._visit(t.item)

    def visit_type_alias_type(self, t: types.TypeAliasType) -> set[str]:
        if False:
            while True:
                i = 10
        return self._visit(types.get_proper_type(t))