from __future__ import annotations
from typing import Final, Iterable, Mapping, Sequence, TypeVar, cast, overload
from mypy.nodes import ARG_STAR, Var
from mypy.state import state
from mypy.types import ANY_STRATEGY, AnyType, BoolTypeQuery, CallableType, DeletedType, ErasedType, FunctionLike, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecFlavor, ParamSpecType, PartialType, ProperType, TrivialSyntheticTypeTranslator, TupleType, Type, TypeAliasType, TypedDictType, TypeType, TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType, flatten_nested_unions, get_proper_type, split_with_prefix_and_suffix
from mypy.typevartuples import split_with_instance
import mypy.type_visitor

@overload
def expand_type(typ: CallableType, env: Mapping[TypeVarId, Type]) -> CallableType:
    if False:
        while True:
            i = 10
    ...

@overload
def expand_type(typ: ProperType, env: Mapping[TypeVarId, Type]) -> ProperType:
    if False:
        i = 10
        return i + 15
    ...

@overload
def expand_type(typ: Type, env: Mapping[TypeVarId, Type]) -> Type:
    if False:
        for i in range(10):
            print('nop')
    ...

def expand_type(typ: Type, env: Mapping[TypeVarId, Type]) -> Type:
    if False:
        return 10
    'Substitute any type variable references in a type given by a type\n    environment.\n    '
    return typ.accept(ExpandTypeVisitor(env))

@overload
def expand_type_by_instance(typ: CallableType, instance: Instance) -> CallableType:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def expand_type_by_instance(typ: ProperType, instance: Instance) -> ProperType:
    if False:
        return 10
    ...

@overload
def expand_type_by_instance(typ: Type, instance: Instance) -> Type:
    if False:
        for i in range(10):
            print('nop')
    ...

def expand_type_by_instance(typ: Type, instance: Instance) -> Type:
    if False:
        i = 10
        return i + 15
    'Substitute type variables in type using values from an Instance.\n    Type variables are considered to be bound by the class declaration.'
    if not instance.args and (not instance.type.has_type_var_tuple_type):
        return typ
    else:
        variables: dict[TypeVarId, Type] = {}
        if instance.type.has_type_var_tuple_type:
            assert instance.type.type_var_tuple_prefix is not None
            assert instance.type.type_var_tuple_suffix is not None
            (args_prefix, args_middle, args_suffix) = split_with_instance(instance)
            (tvars_prefix, tvars_middle, tvars_suffix) = split_with_prefix_and_suffix(tuple(instance.type.defn.type_vars), instance.type.type_var_tuple_prefix, instance.type.type_var_tuple_suffix)
            tvar = tvars_middle[0]
            assert isinstance(tvar, TypeVarTupleType)
            variables = {tvar.id: TupleType(list(args_middle), tvar.tuple_fallback)}
            instance_args = args_prefix + args_suffix
            tvars = tvars_prefix + tvars_suffix
        else:
            tvars = tuple(instance.type.defn.type_vars)
            instance_args = instance.args
        for (binder, arg) in zip(tvars, instance_args):
            assert isinstance(binder, TypeVarLikeType)
            variables[binder.id] = arg
        return expand_type(typ, variables)
F = TypeVar('F', bound=FunctionLike)

def freshen_function_type_vars(callee: F) -> F:
    if False:
        while True:
            i = 10
    'Substitute fresh type variables for generic function type variables.'
    if isinstance(callee, CallableType):
        if not callee.is_generic():
            return cast(F, callee)
        tvs = []
        tvmap: dict[TypeVarId, Type] = {}
        for v in callee.variables:
            tv = v.new_unification_variable(v)
            tvs.append(tv)
            tvmap[v.id] = tv
        fresh = expand_type(callee, tvmap).copy_modified(variables=tvs)
        return cast(F, fresh)
    else:
        assert isinstance(callee, Overloaded)
        fresh_overload = Overloaded([freshen_function_type_vars(item) for item in callee.items])
        return cast(F, fresh_overload)

class HasGenericCallable(BoolTypeQuery):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__(ANY_STRATEGY)

    def visit_callable_type(self, t: CallableType) -> bool:
        if False:
            i = 10
            return i + 15
        return t.is_generic() or super().visit_callable_type(t)
has_generic_callable: Final = HasGenericCallable()
T = TypeVar('T', bound=Type)

def freshen_all_functions_type_vars(t: T) -> T:
    if False:
        return 10
    result: Type
    has_generic_callable.reset()
    if not t.accept(has_generic_callable):
        return t
    else:
        result = t.accept(FreshenCallableVisitor())
        assert isinstance(result, type(t))
        return result

class FreshenCallableVisitor(mypy.type_visitor.TypeTranslator):

    def visit_callable_type(self, t: CallableType) -> Type:
        if False:
            return 10
        result = super().visit_callable_type(t)
        assert isinstance(result, ProperType) and isinstance(result, CallableType)
        return freshen_function_type_vars(result)

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        return t.copy_modified(args=[arg.accept(self) for arg in t.args])

class ExpandTypeVisitor(TrivialSyntheticTypeTranslator):
    """Visitor that substitutes type variables with values."""
    variables: Mapping[TypeVarId, Type]

    def __init__(self, variables: Mapping[TypeVarId, Type]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.variables = variables

    def visit_unbound_type(self, t: UnboundType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_any(self, t: AnyType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_none_type(self, t: NoneType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_uninhabited_type(self, t: UninhabitedType) -> Type:
        if False:
            print('Hello World!')
        return t

    def visit_deleted_type(self, t: DeletedType) -> Type:
        if False:
            i = 10
            return i + 15
        return t

    def visit_erased_type(self, t: ErasedType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        return t

    def visit_instance(self, t: Instance) -> Type:
        if False:
            i = 10
            return i + 15
        args = self.expand_types_with_unpack(list(t.args))
        if t.type.fullname == 'builtins.tuple':
            arg = args[0]
            if isinstance(arg, UnpackType):
                unpacked = get_proper_type(arg.type)
                if isinstance(unpacked, Instance):
                    assert unpacked.type.fullname == 'builtins.tuple'
                    args = list(unpacked.args)
        return t.copy_modified(args=args)

    def visit_type_var(self, t: TypeVarType) -> Type:
        if False:
            print('Hello World!')
        if t.id.raw_id == 0:
            t = t.copy_modified(upper_bound=t.upper_bound.accept(self))
        repl = self.variables.get(t.id, t)
        if isinstance(repl, ProperType) and isinstance(repl, Instance):
            return repl.copy_modified(last_known_value=None)
        return repl

    def visit_param_spec(self, t: ParamSpecType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        repl = self.variables.get(t.id, t.copy_modified(prefix=Parameters([], [], [])))
        if isinstance(repl, ParamSpecType):
            return repl.copy_modified(flavor=t.flavor, prefix=t.prefix.copy_modified(arg_types=self.expand_types(t.prefix.arg_types) + repl.prefix.arg_types, arg_kinds=t.prefix.arg_kinds + repl.prefix.arg_kinds, arg_names=t.prefix.arg_names + repl.prefix.arg_names))
        elif isinstance(repl, Parameters):
            assert t.flavor == ParamSpecFlavor.BARE
            return Parameters(self.expand_types(t.prefix.arg_types) + repl.arg_types, t.prefix.arg_kinds + repl.arg_kinds, t.prefix.arg_names + repl.arg_names, variables=[*t.prefix.variables, *repl.variables])
        else:
            return repl

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        repl = self.variables.get(t.id, t)
        if isinstance(repl, TypeVarTupleType):
            return repl
        raise NotImplementedError

    def visit_unpack_type(self, t: UnpackType) -> Type:
        if False:
            return 10
        return UnpackType(t.type.accept(self))

    def expand_unpack(self, t: UnpackType) -> list[Type]:
        if False:
            print('Hello World!')
        assert isinstance(t.type, TypeVarTupleType)
        repl = get_proper_type(self.variables.get(t.type.id, t.type))
        if isinstance(repl, TupleType):
            return repl.items
        elif isinstance(repl, Instance) and repl.type.fullname == 'builtins.tuple' or isinstance(repl, TypeVarTupleType):
            return [UnpackType(typ=repl)]
        elif isinstance(repl, (AnyType, UninhabitedType)):
            return [UnpackType(t.type.tuple_fallback.copy_modified(args=[repl]))]
        else:
            raise RuntimeError(f'Invalid type replacement to expand: {repl}')

    def visit_parameters(self, t: Parameters) -> Type:
        if False:
            return 10
        return t.copy_modified(arg_types=self.expand_types(t.arg_types))

    def interpolate_args_for_unpack(self, t: CallableType, var_arg: UnpackType) -> list[Type]:
        if False:
            for i in range(10):
                print('nop')
        star_index = t.arg_kinds.index(ARG_STAR)
        prefix = self.expand_types(t.arg_types[:star_index])
        suffix = self.expand_types(t.arg_types[star_index + 1:])
        var_arg_type = get_proper_type(var_arg.type)
        if isinstance(var_arg_type, TupleType):
            expanded_tuple = var_arg_type.accept(self)
            assert isinstance(expanded_tuple, ProperType) and isinstance(expanded_tuple, TupleType)
            expanded_items = expanded_tuple.items
            fallback = var_arg_type.partial_fallback
        else:
            assert isinstance(var_arg_type, TypeVarTupleType)
            fallback = var_arg_type.tuple_fallback
            expanded_items = self.expand_unpack(var_arg)
        new_unpack = UnpackType(TupleType(expanded_items, fallback))
        return prefix + [new_unpack] + suffix

    def visit_callable_type(self, t: CallableType) -> CallableType:
        if False:
            return 10
        param_spec = t.param_spec()
        if param_spec is not None:
            repl = self.variables.get(param_spec.id)
            if isinstance(repl, Parameters):
                return t.copy_modified(arg_types=self.expand_types(t.arg_types[:-2]) + repl.arg_types, arg_kinds=t.arg_kinds[:-2] + repl.arg_kinds, arg_names=t.arg_names[:-2] + repl.arg_names, ret_type=t.ret_type.accept(self), type_guard=t.type_guard.accept(self) if t.type_guard is not None else None, imprecise_arg_kinds=t.imprecise_arg_kinds or repl.imprecise_arg_kinds, variables=[*repl.variables, *t.variables])
            elif isinstance(repl, ParamSpecType):
                prefix = repl.prefix
                clean_repl = repl.copy_modified(prefix=Parameters([], [], []))
                return t.copy_modified(arg_types=self.expand_types(t.arg_types[:-2]) + prefix.arg_types + [clean_repl.with_flavor(ParamSpecFlavor.ARGS), clean_repl.with_flavor(ParamSpecFlavor.KWARGS)], arg_kinds=t.arg_kinds[:-2] + prefix.arg_kinds + t.arg_kinds[-2:], arg_names=t.arg_names[:-2] + prefix.arg_names + t.arg_names[-2:], ret_type=t.ret_type.accept(self), from_concatenate=t.from_concatenate or bool(repl.prefix.arg_types), imprecise_arg_kinds=t.imprecise_arg_kinds or prefix.imprecise_arg_kinds)
        var_arg = t.var_arg()
        needs_normalization = False
        if var_arg is not None and isinstance(var_arg.typ, UnpackType):
            needs_normalization = True
            arg_types = self.interpolate_args_for_unpack(t, var_arg.typ)
        else:
            arg_types = self.expand_types(t.arg_types)
        expanded = t.copy_modified(arg_types=arg_types, ret_type=t.ret_type.accept(self), type_guard=t.type_guard.accept(self) if t.type_guard is not None else None)
        if needs_normalization:
            return expanded.with_normalized_var_args()
        return expanded

    def visit_overloaded(self, t: Overloaded) -> Type:
        if False:
            for i in range(10):
                print('nop')
        items: list[CallableType] = []
        for item in t.items:
            new_item = item.accept(self)
            assert isinstance(new_item, ProperType)
            assert isinstance(new_item, CallableType)
            items.append(new_item)
        return Overloaded(items)

    def expand_types_with_unpack(self, typs: Sequence[Type]) -> list[Type]:
        if False:
            print('Hello World!')
        'Expands a list of types that has an unpack.'
        items: list[Type] = []
        for item in typs:
            if isinstance(item, UnpackType) and isinstance(item.type, TypeVarTupleType):
                items.extend(self.expand_unpack(item))
            else:
                items.append(item.accept(self))
        return items

    def visit_tuple_type(self, t: TupleType) -> Type:
        if False:
            return 10
        items = self.expand_types_with_unpack(t.items)
        if len(items) == 1:
            item = items[0]
            if isinstance(item, UnpackType):
                unpacked = get_proper_type(item.type)
                if isinstance(unpacked, Instance):
                    assert unpacked.type.fullname == 'builtins.tuple'
                    if t.partial_fallback.type.fullname != 'builtins.tuple':
                        return t.partial_fallback.accept(self)
                    return unpacked
        fallback = t.partial_fallback.accept(self)
        assert isinstance(fallback, ProperType) and isinstance(fallback, Instance)
        return t.copy_modified(items=items, fallback=fallback)

    def visit_typeddict_type(self, t: TypedDictType) -> Type:
        if False:
            while True:
                i = 10
        fallback = t.fallback.accept(self)
        assert isinstance(fallback, ProperType) and isinstance(fallback, Instance)
        return t.copy_modified(item_types=self.expand_types(t.items.values()), fallback=fallback)

    def visit_literal_type(self, t: LiteralType) -> Type:
        if False:
            for i in range(10):
                print('nop')
        return t

    def visit_union_type(self, t: UnionType) -> Type:
        if False:
            print('Hello World!')
        expanded = self.expand_types(t.items)
        simplified = UnionType.make_union(remove_trivial(flatten_nested_unions(expanded)), t.line, t.column)
        return get_proper_type(simplified)

    def visit_partial_type(self, t: PartialType) -> Type:
        if False:
            while True:
                i = 10
        return t

    def visit_type_type(self, t: TypeType) -> Type:
        if False:
            print('Hello World!')
        item = t.item.accept(self)
        return TypeType.make_normalized(item)

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            print('Hello World!')
        args = self.expand_types_with_unpack(t.args)
        return t.copy_modified(args=args)

    def expand_types(self, types: Iterable[Type]) -> list[Type]:
        if False:
            for i in range(10):
                print('nop')
        a: list[Type] = []
        for t in types:
            a.append(t.accept(self))
        return a

@overload
def expand_self_type(var: Var, typ: ProperType, replacement: ProperType) -> ProperType:
    if False:
        while True:
            i = 10
    ...

@overload
def expand_self_type(var: Var, typ: Type, replacement: Type) -> Type:
    if False:
        print('Hello World!')
    ...

def expand_self_type(var: Var, typ: Type, replacement: Type) -> Type:
    if False:
        while True:
            i = 10
    'Expand appearances of Self type in a variable type.'
    if var.info.self_type is not None and (not var.is_property):
        return expand_type(typ, {var.info.self_type.id: replacement})
    return typ

def remove_trivial(types: Iterable[Type]) -> list[Type]:
    if False:
        i = 10
        return i + 15
    'Make trivial simplifications on a list of types without calling is_subtype().\n\n    This makes following simplifications:\n        * Remove bottom types (taking into account strict optional setting)\n        * Remove everything else if there is an `object`\n        * Remove strict duplicate types\n    '
    removed_none = False
    new_types = []
    all_types = set()
    for t in types:
        p_t = get_proper_type(t)
        if isinstance(p_t, UninhabitedType):
            continue
        if isinstance(p_t, NoneType) and (not state.strict_optional):
            removed_none = True
            continue
        if isinstance(p_t, Instance) and p_t.type.fullname == 'builtins.object':
            return [p_t]
        if p_t not in all_types:
            new_types.append(t)
            all_types.add(p_t)
    if new_types:
        return new_types
    if removed_none:
        return [NoneType()]
    return [UninhabitedType()]