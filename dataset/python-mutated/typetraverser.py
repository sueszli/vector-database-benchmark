from __future__ import annotations
from typing import Iterable
from mypy_extensions import trait
from mypy.types import AnyType, CallableArgument, CallableType, DeletedType, EllipsisType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, PlaceholderType, RawExpressionType, SyntheticTypeVisitor, TupleType, Type, TypeAliasType, TypedDictType, TypeList, TypeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType

@trait
class TypeTraverserVisitor(SyntheticTypeVisitor[None]):
    """Visitor that traverses all components of a type"""

    def visit_any(self, t: AnyType) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_uninhabited_type(self, t: UninhabitedType) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_none_type(self, t: NoneType) -> None:
        if False:
            return 10
        pass

    def visit_erased_type(self, t: ErasedType) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_deleted_type(self, t: DeletedType) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_type_var(self, t: TypeVarType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_param_spec(self, t: ParamSpecType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_parameters(self, t: Parameters) -> None:
        if False:
            return 10
        self.traverse_types(t.arg_types)

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_literal_type(self, t: LiteralType) -> None:
        if False:
            while True:
                i = 10
        t.fallback.accept(self)

    def visit_instance(self, t: Instance) -> None:
        if False:
            print('Hello World!')
        self.traverse_types(t.args)

    def visit_callable_type(self, t: CallableType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.traverse_types(t.arg_types)
        t.ret_type.accept(self)
        t.fallback.accept(self)

    def visit_tuple_type(self, t: TupleType) -> None:
        if False:
            return 10
        self.traverse_types(t.items)
        t.partial_fallback.accept(self)

    def visit_typeddict_type(self, t: TypedDictType) -> None:
        if False:
            return 10
        self.traverse_types(t.items.values())
        t.fallback.accept(self)

    def visit_union_type(self, t: UnionType) -> None:
        if False:
            return 10
        self.traverse_types(t.items)

    def visit_overloaded(self, t: Overloaded) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.traverse_types(t.items)

    def visit_type_type(self, t: TypeType) -> None:
        if False:
            for i in range(10):
                print('nop')
        t.item.accept(self)

    def visit_callable_argument(self, t: CallableArgument) -> None:
        if False:
            i = 10
            return i + 15
        t.typ.accept(self)

    def visit_unbound_type(self, t: UnboundType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.traverse_types(t.args)

    def visit_type_list(self, t: TypeList) -> None:
        if False:
            print('Hello World!')
        self.traverse_types(t.items)

    def visit_ellipsis_type(self, t: EllipsisType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_placeholder_type(self, t: PlaceholderType) -> None:
        if False:
            i = 10
            return i + 15
        self.traverse_types(t.args)

    def visit_partial_type(self, t: PartialType) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_raw_expression_type(self, t: RawExpressionType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_type_alias_type(self, t: TypeAliasType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.traverse_types(t.args)

    def visit_unpack_type(self, t: UnpackType) -> None:
        if False:
            i = 10
            return i + 15
        t.type.accept(self)

    def traverse_types(self, types: Iterable[Type]) -> None:
        if False:
            i = 10
            return i + 15
        for typ in types:
            typ.accept(self)