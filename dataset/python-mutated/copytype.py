from __future__ import annotations
from typing import Any, cast
from mypy.types import AnyType, CallableType, DeletedType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, ProperType, TupleType, TypeAliasType, TypedDictType, TypeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType
from mypy.type_visitor import TypeVisitor

def copy_type(t: ProperType) -> ProperType:
    if False:
        while True:
            i = 10
    "Create a shallow copy of a type.\n\n    This can be used to mutate the copy with truthiness information.\n\n    Classes compiled with mypyc don't support copy.copy(), so we need\n    a custom implementation.\n    "
    return t.accept(TypeShallowCopier())

class TypeShallowCopier(TypeVisitor[ProperType]):

    def visit_unbound_type(self, t: UnboundType) -> ProperType:
        if False:
            while True:
                i = 10
        return t

    def visit_any(self, t: AnyType) -> ProperType:
        if False:
            print('Hello World!')
        return self.copy_common(t, AnyType(t.type_of_any, t.source_any, t.missing_import_name))

    def visit_none_type(self, t: NoneType) -> ProperType:
        if False:
            i = 10
            return i + 15
        return self.copy_common(t, NoneType())

    def visit_uninhabited_type(self, t: UninhabitedType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        dup = UninhabitedType(t.is_noreturn)
        dup.ambiguous = t.ambiguous
        return self.copy_common(t, dup)

    def visit_erased_type(self, t: ErasedType) -> ProperType:
        if False:
            while True:
                i = 10
        return self.copy_common(t, ErasedType())

    def visit_deleted_type(self, t: DeletedType) -> ProperType:
        if False:
            print('Hello World!')
        return self.copy_common(t, DeletedType(t.source))

    def visit_instance(self, t: Instance) -> ProperType:
        if False:
            return 10
        dup = Instance(t.type, t.args, last_known_value=t.last_known_value)
        dup.invalid = t.invalid
        return self.copy_common(t, dup)

    def visit_type_var(self, t: TypeVarType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return self.copy_common(t, t.copy_modified())

    def visit_param_spec(self, t: ParamSpecType) -> ProperType:
        if False:
            return 10
        dup = ParamSpecType(t.name, t.fullname, t.id, t.flavor, t.upper_bound, t.default, prefix=t.prefix)
        return self.copy_common(t, dup)

    def visit_parameters(self, t: Parameters) -> ProperType:
        if False:
            i = 10
            return i + 15
        dup = Parameters(t.arg_types, t.arg_kinds, t.arg_names, variables=t.variables, is_ellipsis_args=t.is_ellipsis_args)
        return self.copy_common(t, dup)

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> ProperType:
        if False:
            while True:
                i = 10
        dup = TypeVarTupleType(t.name, t.fullname, t.id, t.upper_bound, t.tuple_fallback, t.default)
        return self.copy_common(t, dup)

    def visit_unpack_type(self, t: UnpackType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        dup = UnpackType(t.type)
        return self.copy_common(t, dup)

    def visit_partial_type(self, t: PartialType) -> ProperType:
        if False:
            print('Hello World!')
        return self.copy_common(t, PartialType(t.type, t.var, t.value_type))

    def visit_callable_type(self, t: CallableType) -> ProperType:
        if False:
            return 10
        return self.copy_common(t, t.copy_modified())

    def visit_tuple_type(self, t: TupleType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return self.copy_common(t, TupleType(t.items, t.partial_fallback, implicit=t.implicit))

    def visit_typeddict_type(self, t: TypedDictType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return self.copy_common(t, TypedDictType(t.items, t.required_keys, t.fallback))

    def visit_literal_type(self, t: LiteralType) -> ProperType:
        if False:
            while True:
                i = 10
        return self.copy_common(t, LiteralType(value=t.value, fallback=t.fallback))

    def visit_union_type(self, t: UnionType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return self.copy_common(t, UnionType(t.items))

    def visit_overloaded(self, t: Overloaded) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return self.copy_common(t, Overloaded(items=t.items))

    def visit_type_type(self, t: TypeType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return self.copy_common(t, TypeType(cast(Any, t.item)))

    def visit_type_alias_type(self, t: TypeAliasType) -> ProperType:
        if False:
            return 10
        assert False, 'only ProperTypes supported'

    def copy_common(self, t: ProperType, t2: ProperType) -> ProperType:
        if False:
            print('Hello World!')
        t2.line = t.line
        t2.column = t.column
        t2.can_be_false = t.can_be_false
        t2.can_be_true = t.can_be_true
        return t2