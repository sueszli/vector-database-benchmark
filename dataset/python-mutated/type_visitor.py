"""Type visitor classes.

This module defines the type visitors that are intended to be
subclassed by other code.  They have been separated out into their own
module to ease converting mypy to run under mypyc, since currently
mypyc-extension classes can extend interpreted classes but not the
other way around. Separating them out, then, allows us to compile
types before we can compile everything that uses a TypeVisitor.

The visitors are all re-exported from mypy.types and that is how
other modules refer to them.
"""
from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable, Final, Generic, Iterable, Sequence, TypeVar, cast
from mypy_extensions import mypyc_attr, trait
from mypy.types import AnyType, CallableArgument, CallableType, DeletedType, EllipsisType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, PlaceholderType, RawExpressionType, TupleType, Type, TypeAliasType, TypedDictType, TypeList, TypeType, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType, get_proper_type
T = TypeVar('T')

@trait
@mypyc_attr(allow_interpreted_subclasses=True)
class TypeVisitor(Generic[T]):
    """Visitor class for types (Type subclasses).

    The parameter T is the return type of the visit methods.
    """

    @abstractmethod
    def visit_unbound_type(self, t: UnboundType) -> T:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def visit_any(self, t: AnyType) -> T:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def visit_none_type(self, t: NoneType) -> T:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def visit_uninhabited_type(self, t: UninhabitedType) -> T:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def visit_erased_type(self, t: ErasedType) -> T:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def visit_deleted_type(self, t: DeletedType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_type_var(self, t: TypeVarType) -> T:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def visit_param_spec(self, t: ParamSpecType) -> T:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def visit_parameters(self, t: Parameters) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_type_var_tuple(self, t: TypeVarTupleType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_instance(self, t: Instance) -> T:
        if False:
            return 10
        pass

    @abstractmethod
    def visit_callable_type(self, t: CallableType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_overloaded(self, t: Overloaded) -> T:
        if False:
            return 10
        pass

    @abstractmethod
    def visit_tuple_type(self, t: TupleType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_typeddict_type(self, t: TypedDictType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_literal_type(self, t: LiteralType) -> T:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def visit_union_type(self, t: UnionType) -> T:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def visit_partial_type(self, t: PartialType) -> T:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def visit_type_type(self, t: TypeType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_type_alias_type(self, t: TypeAliasType) -> T:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def visit_unpack_type(self, t: UnpackType) -> T:
        if False:
            print('Hello World!')
        pass

@trait
@mypyc_attr(allow_interpreted_subclasses=True)
class SyntheticTypeVisitor(TypeVisitor[T]):
    """A TypeVisitor that also knows how to visit synthetic AST constructs.

    Not just real types.
    """

    @abstractmethod
    def visit_type_list(self, t: TypeList) -> T:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def visit_callable_argument(self, t: CallableArgument) -> T:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def visit_ellipsis_type(self, t: EllipsisType) -> T:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def visit_raw_expression_type(self, t: RawExpressionType) -> T:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def visit_placeholder_type(self, t: PlaceholderType) -> T:
        if False:
            return 10
        pass

@mypyc_attr(allow_interpreted_subclasses=True)
class TypeTranslator(TypeVisitor[Type]):
    """Identity type transformation.

    Subclass this and override some methods to implement a non-trivial
    transformation.
    """

    def visit_unbound_type(self, t: UnboundType) -> Type:
        if False:
            i = 10
            return i + 15
        return t

    def visit_any(self, t: AnyType) -> Type:
        if False:
            return 10
        return t

    def visit_none_type(self, t: NoneType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_uninhabited_type(self, t: UninhabitedType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_erased_type(self, t: ErasedType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        return t

    def visit_deleted_type(self, t: DeletedType) -> Type:
        if False:
            return 10
        return t

    def visit_instance(self, t: Instance) -> Type:
        if False:
            i = 10
            return i + 15
        last_known_value: LiteralType | None = None
        if t.last_known_value is not None:
            raw_last_known_value = t.last_known_value.accept(self)
            assert isinstance(raw_last_known_value, LiteralType)
            last_known_value = raw_last_known_value
        return Instance(typ=t.type, args=self.translate_types(t.args), line=t.line, column=t.column, last_known_value=last_known_value)

    def visit_type_var(self, t: TypeVarType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_param_spec(self, t: ParamSpecType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_parameters(self, t: Parameters) -> Type:
        if False:
            return 10
        return t.copy_modified(arg_types=self.translate_types(t.arg_types))

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_partial_type(self, t: PartialType) -> Type:
        if False:
            return 10
        return t

    def visit_unpack_type(self, t: UnpackType) -> Type:
        if False:
            while True:
                i = 10
        return UnpackType(t.type.accept(self))

    def visit_callable_type(self, t: CallableType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        return t.copy_modified(arg_types=self.translate_types(t.arg_types), ret_type=t.ret_type.accept(self), variables=self.translate_variables(t.variables))

    def visit_tuple_type(self, t: TupleType) -> Type:
        if False:
            print('Hello World!')
        return TupleType(self.translate_types(t.items), cast(Any, t.partial_fallback.accept(self)), t.line, t.column)

    def visit_typeddict_type(self, t: TypedDictType) -> Type:
        if False:
            i = 10
            return i + 15
        items = {item_name: item_type.accept(self) for (item_name, item_type) in t.items.items()}
        return TypedDictType(items, t.required_keys, cast(Any, t.fallback.accept(self)), t.line, t.column)

    def visit_literal_type(self, t: LiteralType) -> Type:
        if False:
            return 10
        fallback = t.fallback.accept(self)
        assert isinstance(fallback, Instance)
        return LiteralType(value=t.value, fallback=fallback, line=t.line, column=t.column)

    def visit_union_type(self, t: UnionType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        return UnionType(self.translate_types(t.items), t.line, t.column)

    def translate_types(self, types: Iterable[Type]) -> list[Type]:
        if False:
            while True:
                i = 10
        return [t.accept(self) for t in types]

    def translate_variables(self, variables: Sequence[TypeVarLikeType]) -> Sequence[TypeVarLikeType]:
        if False:
            for i in range(10):
                print('nop')
        return variables

    def visit_overloaded(self, t: Overloaded) -> Type:
        if False:
            while True:
                i = 10
        items: list[CallableType] = []
        for item in t.items:
            new = item.accept(self)
            assert isinstance(new, CallableType)
            items.append(new)
        return Overloaded(items=items)

    def visit_type_type(self, t: TypeType) -> Type:
        if False:
            return 10
        return TypeType.make_normalized(t.item.accept(self), line=t.line, column=t.column)

    @abstractmethod
    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        pass

@mypyc_attr(allow_interpreted_subclasses=True)
class TypeQuery(SyntheticTypeVisitor[T]):
    """Visitor for performing queries of types.

    strategy is used to combine results for a series of types,
    common use cases involve a boolean query using `any` or `all`.

    Note: this visitor keeps an internal state (tracks type aliases to avoid
    recursion), so it should *never* be re-used for querying different types,
    create a new visitor instance instead.

    # TODO: check that we don't have existing violations of this rule.
    """

    def __init__(self, strategy: Callable[[list[T]], T]) -> None:
        if False:
            print('Hello World!')
        self.strategy = strategy
        self.seen_aliases: set[TypeAliasType] = set()
        self.skip_alias_target = False

    def visit_unbound_type(self, t: UnboundType) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types(t.args)

    def visit_type_list(self, t: TypeList) -> T:
        if False:
            print('Hello World!')
        return self.query_types(t.items)

    def visit_callable_argument(self, t: CallableArgument) -> T:
        if False:
            print('Hello World!')
        return t.typ.accept(self)

    def visit_any(self, t: AnyType) -> T:
        if False:
            while True:
                i = 10
        return self.strategy([])

    def visit_uninhabited_type(self, t: UninhabitedType) -> T:
        if False:
            return 10
        return self.strategy([])

    def visit_none_type(self, t: NoneType) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.strategy([])

    def visit_erased_type(self, t: ErasedType) -> T:
        if False:
            while True:
                i = 10
        return self.strategy([])

    def visit_deleted_type(self, t: DeletedType) -> T:
        if False:
            print('Hello World!')
        return self.strategy([])

    def visit_type_var(self, t: TypeVarType) -> T:
        if False:
            i = 10
            return i + 15
        return self.query_types([t.upper_bound, t.default] + t.values)

    def visit_param_spec(self, t: ParamSpecType) -> T:
        if False:
            return 10
        return self.query_types([t.upper_bound, t.default, t.prefix])

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> T:
        if False:
            i = 10
            return i + 15
        return self.query_types([t.upper_bound, t.default])

    def visit_unpack_type(self, t: UnpackType) -> T:
        if False:
            return 10
        return self.query_types([t.type])

    def visit_parameters(self, t: Parameters) -> T:
        if False:
            i = 10
            return i + 15
        return self.query_types(t.arg_types)

    def visit_partial_type(self, t: PartialType) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.strategy([])

    def visit_instance(self, t: Instance) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types(t.args)

    def visit_callable_type(self, t: CallableType) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types(t.arg_types + [t.ret_type])

    def visit_tuple_type(self, t: TupleType) -> T:
        if False:
            return 10
        return self.query_types(t.items)

    def visit_typeddict_type(self, t: TypedDictType) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types(t.items.values())

    def visit_raw_expression_type(self, t: RawExpressionType) -> T:
        if False:
            print('Hello World!')
        return self.strategy([])

    def visit_literal_type(self, t: LiteralType) -> T:
        if False:
            while True:
                i = 10
        return self.strategy([])

    def visit_union_type(self, t: UnionType) -> T:
        if False:
            while True:
                i = 10
        return self.query_types(t.items)

    def visit_overloaded(self, t: Overloaded) -> T:
        if False:
            return 10
        return self.query_types(t.items)

    def visit_type_type(self, t: TypeType) -> T:
        if False:
            print('Hello World!')
        return t.item.accept(self)

    def visit_ellipsis_type(self, t: EllipsisType) -> T:
        if False:
            for i in range(10):
                print('nop')
        return self.strategy([])

    def visit_placeholder_type(self, t: PlaceholderType) -> T:
        if False:
            print('Hello World!')
        return self.query_types(t.args)

    def visit_type_alias_type(self, t: TypeAliasType) -> T:
        if False:
            for i in range(10):
                print('nop')
        if t in self.seen_aliases:
            return self.strategy([])
        self.seen_aliases.add(t)
        if self.skip_alias_target:
            return self.query_types(t.args)
        return get_proper_type(t).accept(self)

    def query_types(self, types: Iterable[Type]) -> T:
        if False:
            return 10
        'Perform a query for a list of types using the strategy to combine the results.'
        return self.strategy([t.accept(self) for t in types])
ANY_STRATEGY: Final = 0
ALL_STRATEGY: Final = 1

class BoolTypeQuery(SyntheticTypeVisitor[bool]):
    """Visitor for performing recursive queries of types with a bool result.

    Use TypeQuery if you need non-bool results.

    'strategy' is used to combine results for a series of types. It must
    be ANY_STRATEGY or ALL_STRATEGY.

    Note: This visitor keeps an internal state (tracks type aliases to avoid
    recursion), so it should *never* be re-used for querying different types
    unless you call reset() first.
    """

    def __init__(self, strategy: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.strategy = strategy
        if strategy == ANY_STRATEGY:
            self.default = False
        else:
            assert strategy == ALL_STRATEGY
            self.default = True
        self.seen_aliases: set[TypeAliasType] | None = None
        self.skip_alias_target = False

    def reset(self) -> None:
        if False:
            print('Hello World!')
        'Clear mutable state (but preserve strategy).\n\n        This *must* be called if you want to reuse the visitor.\n        '
        self.seen_aliases = None

    def visit_unbound_type(self, t: UnboundType) -> bool:
        if False:
            print('Hello World!')
        return self.query_types(t.args)

    def visit_type_list(self, t: TypeList) -> bool:
        if False:
            i = 10
            return i + 15
        return self.query_types(t.items)

    def visit_callable_argument(self, t: CallableArgument) -> bool:
        if False:
            return 10
        return t.typ.accept(self)

    def visit_any(self, t: AnyType) -> bool:
        if False:
            i = 10
            return i + 15
        return self.default

    def visit_uninhabited_type(self, t: UninhabitedType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.default

    def visit_none_type(self, t: NoneType) -> bool:
        if False:
            return 10
        return self.default

    def visit_erased_type(self, t: ErasedType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.default

    def visit_deleted_type(self, t: DeletedType) -> bool:
        if False:
            return 10
        return self.default

    def visit_type_var(self, t: TypeVarType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types([t.upper_bound, t.default] + t.values)

    def visit_param_spec(self, t: ParamSpecType) -> bool:
        if False:
            print('Hello World!')
        return self.query_types([t.upper_bound, t.default])

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> bool:
        if False:
            i = 10
            return i + 15
        return self.query_types([t.upper_bound, t.default])

    def visit_unpack_type(self, t: UnpackType) -> bool:
        if False:
            return 10
        return self.query_types([t.type])

    def visit_parameters(self, t: Parameters) -> bool:
        if False:
            return 10
        return self.query_types(t.arg_types)

    def visit_partial_type(self, t: PartialType) -> bool:
        if False:
            while True:
                i = 10
        return self.default

    def visit_instance(self, t: Instance) -> bool:
        if False:
            while True:
                i = 10
        return self.query_types(t.args)

    def visit_callable_type(self, t: CallableType) -> bool:
        if False:
            print('Hello World!')
        args = self.query_types(t.arg_types)
        ret = t.ret_type.accept(self)
        if self.strategy == ANY_STRATEGY:
            return args or ret
        else:
            return args and ret

    def visit_tuple_type(self, t: TupleType) -> bool:
        if False:
            return 10
        return self.query_types(t.items)

    def visit_typeddict_type(self, t: TypedDictType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types(list(t.items.values()))

    def visit_raw_expression_type(self, t: RawExpressionType) -> bool:
        if False:
            while True:
                i = 10
        return self.default

    def visit_literal_type(self, t: LiteralType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.default

    def visit_union_type(self, t: UnionType) -> bool:
        if False:
            return 10
        return self.query_types(t.items)

    def visit_overloaded(self, t: Overloaded) -> bool:
        if False:
            while True:
                i = 10
        return self.query_types(t.items)

    def visit_type_type(self, t: TypeType) -> bool:
        if False:
            i = 10
            return i + 15
        return t.item.accept(self)

    def visit_ellipsis_type(self, t: EllipsisType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.default

    def visit_placeholder_type(self, t: PlaceholderType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.query_types(t.args)

    def visit_type_alias_type(self, t: TypeAliasType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self.seen_aliases is None:
            self.seen_aliases = set()
        elif t in self.seen_aliases:
            return self.default
        self.seen_aliases.add(t)
        if self.skip_alias_target:
            return self.query_types(t.args)
        return get_proper_type(t).accept(self)

    def query_types(self, types: list[Type] | tuple[Type, ...]) -> bool:
        if False:
            print('Hello World!')
        'Perform a query for a sequence of types using the strategy to combine the results.'
        if isinstance(types, list):
            if self.strategy == ANY_STRATEGY:
                return any((t.accept(self) for t in types))
            else:
                return all((t.accept(self) for t in types))
        elif self.strategy == ANY_STRATEGY:
            return any((t.accept(self) for t in types))
        else:
            return all((t.accept(self) for t in types))