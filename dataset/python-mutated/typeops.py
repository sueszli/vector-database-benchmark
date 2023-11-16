"""Miscellaneous type operations and helpers for use during type checking.

NOTE: These must not be accessed from mypy.nodes or mypy.types to avoid import
      cycles. These must not be called from the semantic analysis main pass
      since these may assume that MROs are ready.
"""
from __future__ import annotations
import itertools
from typing import Any, Iterable, List, Sequence, TypeVar, cast
from mypy.copytype import copy_type
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import ARG_POS, ARG_STAR, ARG_STAR2, SYMBOL_FUNCBASE_TYPES, Decorator, Expression, FuncBase, FuncDef, FuncItem, OverloadedFuncDef, StrExpr, TypeInfo, Var
from mypy.state import state
from mypy.types import ENUM_REMOVED_PROPS, AnyType, CallableType, ExtraAttrs, FormalArgument, FunctionLike, Instance, LiteralType, NoneType, NormalizedCallableType, Overloaded, Parameters, ParamSpecType, PartialType, ProperType, TupleType, Type, TypeAliasType, TypedDictType, TypeOfAny, TypeQuery, TypeType, TypeVarLikeType, TypeVarTupleType, TypeVarType, UninhabitedType, UnionType, UnpackType, flatten_nested_unions, get_proper_type, get_proper_types
from mypy.typevars import fill_typevars

def is_recursive_pair(s: Type, t: Type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Is this a pair of recursive types?\n\n    There may be more cases, and we may be forced to use e.g. has_recursive_types()\n    here, but this function is called in very hot code, so we try to keep it simple\n    and return True only in cases we know may have problems.\n    '
    if isinstance(s, TypeAliasType) and s.is_recursive:
        return isinstance(get_proper_type(t), (Instance, UnionType)) or (isinstance(t, TypeAliasType) and t.is_recursive) or isinstance(get_proper_type(s), TupleType)
    if isinstance(t, TypeAliasType) and t.is_recursive:
        return isinstance(get_proper_type(s), (Instance, UnionType)) or (isinstance(s, TypeAliasType) and s.is_recursive) or isinstance(get_proper_type(t), TupleType)
    return False

def tuple_fallback(typ: TupleType) -> Instance:
    if False:
        while True:
            i = 10
    'Return fallback type for a tuple.'
    from mypy.join import join_type_list
    info = typ.partial_fallback.type
    if info.fullname != 'builtins.tuple':
        return typ.partial_fallback
    items = []
    for item in typ.items:
        if isinstance(item, UnpackType):
            unpacked_type = get_proper_type(item.type)
            if isinstance(unpacked_type, TypeVarTupleType):
                unpacked_type = get_proper_type(unpacked_type.upper_bound)
            if isinstance(unpacked_type, Instance) and unpacked_type.type.fullname == 'builtins.tuple':
                items.append(unpacked_type.args[0])
            else:
                raise NotImplementedError
        else:
            items.append(item)
    return Instance(info, [join_type_list(items)], extra_attrs=typ.partial_fallback.extra_attrs)

def get_self_type(func: CallableType, default_self: Instance | TupleType) -> Type | None:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(get_proper_type(func.ret_type), UninhabitedType):
        return func.ret_type
    elif func.arg_types and func.arg_types[0] != default_self and (func.arg_kinds[0] == ARG_POS):
        return func.arg_types[0]
    else:
        return None

def type_object_type_from_function(signature: FunctionLike, info: TypeInfo, def_info: TypeInfo, fallback: Instance, is_new: bool) -> FunctionLike:
    if False:
        while True:
            i = 10
    default_self = fill_typevars(info)
    if not is_new and (not info.is_newtype):
        orig_self_types = [get_self_type(it, default_self) for it in signature.items]
    else:
        orig_self_types = [None] * len(signature.items)
    signature = bind_self(signature, original_type=default_self, is_classmethod=is_new)
    signature = cast(FunctionLike, map_type_from_supertype(signature, info, def_info))
    special_sig: str | None = None
    if def_info.fullname == 'builtins.dict':
        special_sig = 'dict'
    if isinstance(signature, CallableType):
        return class_callable(signature, info, fallback, special_sig, is_new, orig_self_types[0])
    else:
        assert isinstance(signature, Overloaded)
        items: list[CallableType] = []
        for (item, orig_self) in zip(signature.items, orig_self_types):
            items.append(class_callable(item, info, fallback, special_sig, is_new, orig_self))
        return Overloaded(items)

def class_callable(init_type: CallableType, info: TypeInfo, type_type: Instance, special_sig: str | None, is_new: bool, orig_self_type: Type | None=None) -> CallableType:
    if False:
        i = 10
        return i + 15
    'Create a type object type based on the signature of __init__.'
    variables: list[TypeVarLikeType] = []
    variables.extend(info.defn.type_vars)
    variables.extend(init_type.variables)
    from mypy.subtypes import is_subtype
    init_ret_type = get_proper_type(init_type.ret_type)
    orig_self_type = get_proper_type(orig_self_type)
    default_ret_type = fill_typevars(info)
    explicit_type = init_ret_type if is_new else orig_self_type
    if isinstance(explicit_type, (Instance, TupleType, UninhabitedType)) and isinstance(default_ret_type, Instance) and (not default_ret_type.type.is_protocol) and is_subtype(explicit_type, default_ret_type, ignore_type_params=True):
        ret_type: Type = explicit_type
    else:
        ret_type = default_ret_type
    callable_type = init_type.copy_modified(ret_type=ret_type, fallback=type_type, name=None, variables=variables, special_sig=special_sig)
    c = callable_type.with_name(info.name)
    return c

def map_type_from_supertype(typ: Type, sub_info: TypeInfo, super_info: TypeInfo) -> Type:
    if False:
        while True:
            i = 10
    'Map type variables in a type defined in a supertype context to be valid\n    in the subtype context. Assume that the result is unique; if more than\n    one type is possible, return one of the alternatives.\n\n    For example, assume\n\n      class D(Generic[S]): ...\n      class C(D[E[T]], Generic[T]): ...\n\n    Now S in the context of D would be mapped to E[T] in the context of C.\n    '
    inst_type = fill_typevars(sub_info)
    if isinstance(inst_type, TupleType):
        inst_type = tuple_fallback(inst_type)
    inst_type = map_instance_to_supertype(inst_type, super_info)
    return expand_type_by_instance(typ, inst_type)

def supported_self_type(typ: ProperType, allow_callable: bool=True) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Is this a supported kind of explicit self-types?\n\n    Currently, this means an X or Type[X], where X is an instance or\n    a type variable with an instance upper bound.\n    '
    if isinstance(typ, TypeType):
        return supported_self_type(typ.item)
    if allow_callable and isinstance(typ, CallableType):
        return True
    return isinstance(typ, TypeVarType) or (isinstance(typ, Instance) and typ != fill_typevars(typ.type))
F = TypeVar('F', bound=FunctionLike)

def bind_self(method: F, original_type: Type | None=None, is_classmethod: bool=False) -> F:
    if False:
        i = 10
        return i + 15
    'Return a copy of `method`, with the type of its first parameter (usually\n    self or cls) bound to original_type.\n\n    If the type of `self` is a generic type (T, or Type[T] for classmethods),\n    instantiate every occurrence of type with original_type in the rest of the\n    signature and in the return type.\n\n    original_type is the type of E in the expression E.copy(). It is None in\n    compatibility checks. In this case we treat it as the erasure of the\n    declared type of self.\n\n    This way we can express "the type of self". For example:\n\n    T = TypeVar(\'T\', bound=\'A\')\n    class A:\n        def copy(self: T) -> T: ...\n\n    class B(A): pass\n\n    b = B().copy()  # type: B\n\n    '
    if isinstance(method, Overloaded):
        return cast(F, Overloaded([bind_self(c, original_type, is_classmethod) for c in method.items]))
    assert isinstance(method, CallableType)
    func = method
    if not func.arg_types:
        return cast(F, func)
    if func.arg_kinds[0] == ARG_STAR:
        return cast(F, func)
    self_param_type = get_proper_type(func.arg_types[0])
    variables: Sequence[TypeVarLikeType]
    allow_callable = func.name is None or not func.name.startswith('__call__ of')
    if func.variables and supported_self_type(self_param_type, allow_callable=allow_callable):
        from mypy.infer import infer_type_arguments
        if original_type is None:
            original_type = erase_to_bound(self_param_type)
        original_type = get_proper_type(original_type)
        self_ids = {tv.id for tv in get_all_type_vars(self_param_type)}
        self_vars = [tv for tv in func.variables if tv.id in self_ids]
        typeargs = infer_type_arguments(self_vars, self_param_type, original_type, is_supertype=True)
        if is_classmethod and any((isinstance(get_proper_type(t), UninhabitedType) for t in typeargs)) and isinstance(original_type, (Instance, TypeVarType, TupleType)):
            typeargs = infer_type_arguments(self_vars, self_param_type, TypeType(original_type), is_supertype=True)
        to_apply = [t if t is not None else UninhabitedType() for t in typeargs]
        func = expand_type(func, {tv.id: arg for (tv, arg) in zip(self_vars, to_apply)})
        variables = [v for v in func.variables if v not in self_vars]
    else:
        variables = func.variables
    original_type = get_proper_type(original_type)
    if isinstance(original_type, CallableType) and original_type.is_type_obj():
        original_type = TypeType.make_normalized(original_type.ret_type)
    res = func.copy_modified(arg_types=func.arg_types[1:], arg_kinds=func.arg_kinds[1:], arg_names=func.arg_names[1:], variables=variables, bound_args=[original_type])
    return cast(F, res)

def erase_to_bound(t: Type) -> Type:
    if False:
        i = 10
        return i + 15
    t = get_proper_type(t)
    if isinstance(t, TypeVarType):
        return t.upper_bound
    if isinstance(t, TypeType):
        if isinstance(t.item, TypeVarType):
            return TypeType.make_normalized(t.item.upper_bound)
    return t

def callable_corresponding_argument(typ: NormalizedCallableType | Parameters, model: FormalArgument) -> FormalArgument | None:
    if False:
        while True:
            i = 10
    'Return the argument a function that corresponds to `model`'
    by_name = typ.argument_by_name(model.name)
    by_pos = typ.argument_by_position(model.pos)
    if by_name is None and by_pos is None:
        return None
    if by_name is not None and by_pos is not None:
        if by_name == by_pos:
            return by_name
        from mypy.subtypes import is_equivalent
        if not (by_name.required or by_pos.required) and by_pos.name is None and (by_name.pos is None) and is_equivalent(by_name.typ, by_pos.typ):
            return FormalArgument(by_name.name, by_pos.pos, by_name.typ, False)
    return by_name if by_name is not None else by_pos

def simple_literal_type(t: ProperType | None) -> Instance | None:
    if False:
        while True:
            i = 10
    'Extract the underlying fallback Instance type for a simple Literal'
    if isinstance(t, Instance) and t.last_known_value is not None:
        t = t.last_known_value
    if isinstance(t, LiteralType):
        return t.fallback
    return None

def is_simple_literal(t: ProperType) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(t, LiteralType):
        return t.fallback.type.is_enum or t.fallback.type.fullname == 'builtins.str'
    if isinstance(t, Instance):
        return t.last_known_value is not None and isinstance(t.last_known_value.value, str)
    return False

def make_simplified_union(items: Sequence[Type], line: int=-1, column: int=-1, *, keep_erased: bool=False, contract_literals: bool=True) -> ProperType:
    if False:
        print('Hello World!')
    'Build union type with redundant union items removed.\n\n    If only a single item remains, this may return a non-union type.\n\n    Examples:\n\n    * [int, str] -> Union[int, str]\n    * [int, object] -> object\n    * [int, int] -> int\n    * [int, Any] -> Union[int, Any] (Any types are not simplified away!)\n    * [Any, Any] -> Any\n    * [int, Union[bytes, str]] -> Union[int, bytes, str]\n\n    Note: This must NOT be used during semantic analysis, since TypeInfos may not\n          be fully initialized.\n\n    The keep_erased flag is used for type inference against union types\n    containing type variables. If set to True, keep all ErasedType items.\n\n    The contract_literals flag indicates whether we need to contract literal types\n    back into a sum type. Set it to False when called by try_expanding_sum_type_\n    to_union().\n    '
    items = flatten_nested_unions(items)
    if len(items) == 1:
        return get_proper_type(items[0])
    simplified_set: Sequence[Type] = _remove_redundant_union_items(items, keep_erased)
    if contract_literals and sum((isinstance(get_proper_type(item), LiteralType) for item in simplified_set)) > 1:
        simplified_set = try_contracting_literals_in_union(simplified_set)
    result = get_proper_type(UnionType.make_union(simplified_set, line, column))
    nitems = len(items)
    if nitems > 1 and (nitems > 2 or not (type(items[0]) is NoneType or type(items[1]) is NoneType)):
        extra_attrs_set: set[ExtraAttrs] | None = None
        for item in items:
            instance = try_getting_instance_fallback(item)
            if instance and instance.extra_attrs:
                if extra_attrs_set is None:
                    extra_attrs_set = {instance.extra_attrs}
                else:
                    extra_attrs_set.add(instance.extra_attrs)
        if extra_attrs_set is not None and len(extra_attrs_set) > 1:
            fallback = try_getting_instance_fallback(result)
            if fallback:
                fallback.extra_attrs = None
    return result

def _remove_redundant_union_items(items: list[Type], keep_erased: bool) -> list[Type]:
    if False:
        i = 10
        return i + 15
    from mypy.subtypes import is_proper_subtype
    for _direction in range(2):
        new_items: list[Type] = []
        seen: dict[ProperType, int] = {}
        unduplicated_literal_fallbacks: set[Instance] | None = None
        for ti in items:
            proper_ti = get_proper_type(ti)
            if isinstance(proper_ti, UninhabitedType):
                continue
            duplicate_index = -1
            if proper_ti in seen:
                duplicate_index = seen[proper_ti]
            elif isinstance(proper_ti, LiteralType) and unduplicated_literal_fallbacks is not None and (proper_ti.fallback in unduplicated_literal_fallbacks):
                pass
            else:
                for (j, tj) in enumerate(new_items):
                    tj = get_proper_type(tj)
                    if isinstance(tj, Instance) and tj.last_known_value is not None and (not (isinstance(proper_ti, Instance) and tj.last_known_value == proper_ti.last_known_value)):
                        continue
                    if is_proper_subtype(ti, tj, keep_erased_types=keep_erased, ignore_promotions=True):
                        duplicate_index = j
                        break
            if duplicate_index != -1:
                orig_item = new_items[duplicate_index]
                if not orig_item.can_be_true and ti.can_be_true:
                    new_items[duplicate_index] = true_or_false(orig_item)
                elif not orig_item.can_be_false and ti.can_be_false:
                    new_items[duplicate_index] = true_or_false(orig_item)
            else:
                seen[proper_ti] = len(new_items)
                new_items.append(ti)
                if isinstance(proper_ti, LiteralType):
                    if unduplicated_literal_fallbacks is None:
                        unduplicated_literal_fallbacks = set()
                    unduplicated_literal_fallbacks.add(proper_ti.fallback)
        items = new_items
        if len(items) <= 1:
            break
        items.reverse()
    return items

def _get_type_special_method_bool_ret_type(t: Type) -> Type | None:
    if False:
        while True:
            i = 10
    t = get_proper_type(t)
    if isinstance(t, Instance):
        bool_method = t.type.get('__bool__')
        if bool_method:
            callee = get_proper_type(bool_method.type)
            if isinstance(callee, CallableType):
                return callee.ret_type
    return None

def true_only(t: Type) -> ProperType:
    if False:
        i = 10
        return i + 15
    '\n    Restricted version of t with only True-ish values\n    '
    t = get_proper_type(t)
    if not t.can_be_true:
        return UninhabitedType(line=t.line, column=t.column)
    elif not t.can_be_false:
        return t
    elif isinstance(t, UnionType):
        new_items = [true_only(item) for item in t.items]
        can_be_true_items = [item for item in new_items if item.can_be_true]
        return make_simplified_union(can_be_true_items, line=t.line, column=t.column)
    else:
        ret_type = _get_type_special_method_bool_ret_type(t)
        if ret_type and (not ret_type.can_be_true):
            return UninhabitedType(line=t.line, column=t.column)
        new_t = copy_type(t)
        new_t.can_be_false = False
        return new_t

def false_only(t: Type) -> ProperType:
    if False:
        while True:
            i = 10
    '\n    Restricted version of t with only False-ish values\n    '
    t = get_proper_type(t)
    if not t.can_be_false:
        if state.strict_optional:
            return UninhabitedType(line=t.line)
        else:
            return NoneType(line=t.line)
    elif not t.can_be_true:
        return t
    elif isinstance(t, UnionType):
        new_items = [false_only(item) for item in t.items]
        can_be_false_items = [item for item in new_items if item.can_be_false]
        return make_simplified_union(can_be_false_items, line=t.line, column=t.column)
    else:
        ret_type = _get_type_special_method_bool_ret_type(t)
        if ret_type and (not ret_type.can_be_false):
            return UninhabitedType(line=t.line)
        new_t = copy_type(t)
        new_t.can_be_true = False
        return new_t

def true_or_false(t: Type) -> ProperType:
    if False:
        return 10
    '\n    Unrestricted version of t with both True-ish and False-ish values\n    '
    t = get_proper_type(t)
    if isinstance(t, UnionType):
        new_items = [true_or_false(item) for item in t.items]
        return make_simplified_union(new_items, line=t.line, column=t.column)
    new_t = copy_type(t)
    new_t.can_be_true = new_t.can_be_true_default()
    new_t.can_be_false = new_t.can_be_false_default()
    return new_t

def erase_def_to_union_or_bound(tdef: TypeVarLikeType) -> Type:
    if False:
        return 10
    if isinstance(tdef, ParamSpecType):
        return AnyType(TypeOfAny.from_error)
    if isinstance(tdef, TypeVarType) and tdef.values:
        return make_simplified_union(tdef.values)
    else:
        return tdef.upper_bound

def erase_to_union_or_bound(typ: TypeVarType) -> ProperType:
    if False:
        for i in range(10):
            print('nop')
    if typ.values:
        return make_simplified_union(typ.values)
    else:
        return get_proper_type(typ.upper_bound)

def function_type(func: FuncBase, fallback: Instance) -> FunctionLike:
    if False:
        i = 10
        return i + 15
    if func.type:
        assert isinstance(func.type, FunctionLike)
        return func.type
    elif isinstance(func, FuncItem):
        return callable_type(func, fallback)
    else:
        assert isinstance(func, OverloadedFuncDef)
        any_type = AnyType(TypeOfAny.from_error)
        dummy = CallableType([any_type, any_type], [ARG_STAR, ARG_STAR2], [None, None], any_type, fallback, line=func.line, is_ellipsis_args=True)
        return Overloaded([dummy])

def callable_type(fdef: FuncItem, fallback: Instance, ret_type: Type | None=None) -> CallableType:
    if False:
        for i in range(10):
            print('nop')
    if fdef.info and (not fdef.is_static or fdef.name == '__new__') and fdef.arg_names:
        self_type: Type = fill_typevars(fdef.info)
        if fdef.is_class or fdef.name == '__new__':
            self_type = TypeType.make_normalized(self_type)
        args = [self_type] + [AnyType(TypeOfAny.unannotated)] * (len(fdef.arg_names) - 1)
    else:
        args = [AnyType(TypeOfAny.unannotated)] * len(fdef.arg_names)
    return CallableType(args, fdef.arg_kinds, fdef.arg_names, ret_type or AnyType(TypeOfAny.unannotated), fallback, name=fdef.name, line=fdef.line, column=fdef.column, implicit=True, definition=fdef if isinstance(fdef, FuncDef) else None)

def try_getting_str_literals(expr: Expression, typ: Type) -> list[str] | None:
    if False:
        i = 10
        return i + 15
    "If the given expression or type corresponds to a string literal\n    or a union of string literals, returns a list of the underlying strings.\n    Otherwise, returns None.\n\n    Specifically, this function is guaranteed to return a list with\n    one or more strings if one of the following is true:\n\n    1. 'expr' is a StrExpr\n    2. 'typ' is a LiteralType containing a string\n    3. 'typ' is a UnionType containing only LiteralType of strings\n    "
    if isinstance(expr, StrExpr):
        return [expr.value]
    return try_getting_str_literals_from_type(typ)

def try_getting_str_literals_from_type(typ: Type) -> list[str] | None:
    if False:
        print('Hello World!')
    'If the given expression or type corresponds to a string Literal\n    or a union of string Literals, returns a list of the underlying strings.\n    Otherwise, returns None.\n\n    For example, if we had the type \'Literal["foo", "bar"]\' as input, this function\n    would return a list of strings ["foo", "bar"].\n    '
    return try_getting_literals_from_type(typ, str, 'builtins.str')

def try_getting_int_literals_from_type(typ: Type) -> list[int] | None:
    if False:
        i = 10
        return i + 15
    "If the given expression or type corresponds to an int Literal\n    or a union of int Literals, returns a list of the underlying ints.\n    Otherwise, returns None.\n\n    For example, if we had the type 'Literal[1, 2, 3]' as input, this function\n    would return a list of ints [1, 2, 3].\n    "
    return try_getting_literals_from_type(typ, int, 'builtins.int')
T = TypeVar('T')

def try_getting_literals_from_type(typ: Type, target_literal_type: type[T], target_fullname: str) -> list[T] | None:
    if False:
        i = 10
        return i + 15
    'If the given expression or type corresponds to a Literal or\n    union of Literals where the underlying values correspond to the given\n    target type, returns a list of those underlying values. Otherwise,\n    returns None.\n    '
    typ = get_proper_type(typ)
    if isinstance(typ, Instance) and typ.last_known_value is not None:
        possible_literals: list[Type] = [typ.last_known_value]
    elif isinstance(typ, UnionType):
        possible_literals = list(typ.items)
    else:
        possible_literals = [typ]
    literals: list[T] = []
    for lit in get_proper_types(possible_literals):
        if isinstance(lit, LiteralType) and lit.fallback.type.fullname == target_fullname:
            val = lit.value
            if isinstance(val, target_literal_type):
                literals.append(val)
            else:
                return None
        else:
            return None
    return literals

def is_literal_type_like(t: Type | None) -> bool:
    if False:
        print('Hello World!')
    "Returns 'true' if the given type context is potentially either a LiteralType,\n    a Union of LiteralType, or something similar.\n    "
    t = get_proper_type(t)
    if t is None:
        return False
    elif isinstance(t, LiteralType):
        return True
    elif isinstance(t, UnionType):
        return any((is_literal_type_like(item) for item in t.items))
    elif isinstance(t, TypeVarType):
        return is_literal_type_like(t.upper_bound) or any((is_literal_type_like(item) for item in t.values))
    else:
        return False

def is_singleton_type(typ: Type) -> bool:
    if False:
        while True:
            i = 10
    'Returns \'true\' if this type is a "singleton type" -- if there exists\n    exactly only one runtime value associated with this type.\n\n    That is, given two values \'a\' and \'b\' that have the same type \'t\',\n    \'is_singleton_type(t)\' returns True if and only if the expression \'a is b\' is\n    always true.\n\n    Currently, this returns True when given NoneTypes, enum LiteralTypes,\n    enum types with a single value and ... (Ellipses).\n\n    Note that other kinds of LiteralTypes cannot count as singleton types. For\n    example, suppose we do \'a = 100000 + 1\' and \'b = 100001\'. It is not guaranteed\n    that \'a is b\' will always be true -- some implementations of Python will end up\n    constructing two distinct instances of 100001.\n    '
    typ = get_proper_type(typ)
    return typ.is_singleton_type()

def try_expanding_sum_type_to_union(typ: Type, target_fullname: str) -> ProperType:
    if False:
        while True:
            i = 10
    "Attempts to recursively expand any enum Instances with the given target_fullname\n    into a Union of all of its component LiteralTypes.\n\n    For example, if we have:\n\n        class Color(Enum):\n            RED = 1\n            BLUE = 2\n            YELLOW = 3\n\n        class Status(Enum):\n            SUCCESS = 1\n            FAILURE = 2\n            UNKNOWN = 3\n\n    ...and if we call `try_expanding_enum_to_union(Union[Color, Status], 'module.Color')`,\n    this function will return Literal[Color.RED, Color.BLUE, Color.YELLOW, Status].\n    "
    typ = get_proper_type(typ)
    if isinstance(typ, UnionType):
        items = [try_expanding_sum_type_to_union(item, target_fullname) for item in typ.relevant_items()]
        return make_simplified_union(items, contract_literals=False)
    elif isinstance(typ, Instance) and typ.type.fullname == target_fullname:
        if typ.type.is_enum:
            new_items = []
            for (name, symbol) in typ.type.names.items():
                if not isinstance(symbol.node, Var):
                    continue
                if name in ENUM_REMOVED_PROPS:
                    continue
                new_items.append(LiteralType(name, typ))
            return make_simplified_union(new_items, contract_literals=False)
        elif typ.type.fullname == 'builtins.bool':
            return make_simplified_union([LiteralType(True, typ), LiteralType(False, typ)], contract_literals=False)
    return typ

def try_contracting_literals_in_union(types: Sequence[Type]) -> list[ProperType]:
    if False:
        print('Hello World!')
    'Contracts any literal types back into a sum type if possible.\n\n    Will replace the first instance of the literal with the sum type and\n    remove all others.\n\n    If we call `try_contracting_union(Literal[Color.RED, Color.BLUE, Color.YELLOW])`,\n    this function will return Color.\n\n    We also treat `Literal[True, False]` as `bool`.\n    '
    proper_types = [get_proper_type(typ) for typ in types]
    sum_types: dict[str, tuple[set[Any], list[int]]] = {}
    marked_for_deletion = set()
    for (idx, typ) in enumerate(proper_types):
        if isinstance(typ, LiteralType):
            fullname = typ.fallback.type.fullname
            if typ.fallback.type.is_enum or isinstance(typ.value, bool):
                if fullname not in sum_types:
                    sum_types[fullname] = (set(typ.fallback.get_enum_values()) if typ.fallback.type.is_enum else {True, False}, [])
                (literals, indexes) = sum_types[fullname]
                literals.discard(typ.value)
                indexes.append(idx)
                if not literals:
                    (first, *rest) = indexes
                    proper_types[first] = typ.fallback
                    marked_for_deletion |= set(rest)
    return list(itertools.compress(proper_types, [i not in marked_for_deletion for i in range(len(proper_types))]))

def coerce_to_literal(typ: Type) -> Type:
    if False:
        for i in range(10):
            print('nop')
    'Recursively converts any Instances that have a last_known_value or are\n    instances of enum types with a single value into the corresponding LiteralType.\n    '
    original_type = typ
    typ = get_proper_type(typ)
    if isinstance(typ, UnionType):
        new_items = [coerce_to_literal(item) for item in typ.items]
        return UnionType.make_union(new_items)
    elif isinstance(typ, Instance):
        if typ.last_known_value:
            return typ.last_known_value
        elif typ.type.is_enum:
            enum_values = typ.get_enum_values()
            if len(enum_values) == 1:
                return LiteralType(value=enum_values[0], fallback=typ)
    return original_type

def get_type_vars(tp: Type) -> list[TypeVarType]:
    if False:
        return 10
    return cast('list[TypeVarType]', tp.accept(TypeVarExtractor()))

def get_all_type_vars(tp: Type) -> list[TypeVarLikeType]:
    if False:
        for i in range(10):
            print('nop')
    return tp.accept(TypeVarExtractor(include_all=True))

class TypeVarExtractor(TypeQuery[List[TypeVarLikeType]]):

    def __init__(self, include_all: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(self._merge)
        self.include_all = include_all

    def _merge(self, iter: Iterable[list[TypeVarLikeType]]) -> list[TypeVarLikeType]:
        if False:
            for i in range(10):
                print('nop')
        out = []
        for item in iter:
            out.extend(item)
        return out

    def visit_type_var(self, t: TypeVarType) -> list[TypeVarLikeType]:
        if False:
            i = 10
            return i + 15
        return [t]

    def visit_param_spec(self, t: ParamSpecType) -> list[TypeVarLikeType]:
        if False:
            for i in range(10):
                print('nop')
        return [t] if self.include_all else []

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> list[TypeVarLikeType]:
        if False:
            print('Hello World!')
        return [t] if self.include_all else []

def custom_special_method(typ: Type, name: str, check_all: bool=False) -> bool:
    if False:
        i = 10
        return i + 15
    'Does this type have a custom special method such as __format__() or __eq__()?\n\n    If check_all is True ensure all items of a union have a custom method, not just some.\n    '
    typ = get_proper_type(typ)
    if isinstance(typ, Instance):
        method = typ.type.get(name)
        if method and isinstance(method.node, (SYMBOL_FUNCBASE_TYPES, Decorator, Var)):
            if method.node.info:
                return not method.node.info.fullname.startswith(('builtins.', 'typing.'))
        return False
    if isinstance(typ, UnionType):
        if check_all:
            return all((custom_special_method(t, name, check_all) for t in typ.items))
        return any((custom_special_method(t, name) for t in typ.items))
    if isinstance(typ, TupleType):
        return custom_special_method(tuple_fallback(typ), name, check_all)
    if isinstance(typ, FunctionLike) and typ.is_type_obj():
        return custom_special_method(typ.fallback, name, check_all)
    if isinstance(typ, AnyType):
        return True
    return False

def separate_union_literals(t: UnionType) -> tuple[Sequence[LiteralType], Sequence[Type]]:
    if False:
        return 10
    'Separate literals from other members in a union type.'
    literal_items = []
    union_items = []
    for item in t.items:
        proper = get_proper_type(item)
        if isinstance(proper, LiteralType):
            literal_items.append(proper)
        else:
            union_items.append(item)
    return (literal_items, union_items)

def try_getting_instance_fallback(typ: Type) -> Instance | None:
    if False:
        while True:
            i = 10
    'Returns the Instance fallback for this type if one exists or None.'
    typ = get_proper_type(typ)
    if isinstance(typ, Instance):
        return typ
    elif isinstance(typ, LiteralType):
        return typ.fallback
    elif isinstance(typ, NoneType):
        return None
    elif isinstance(typ, FunctionLike):
        return typ.fallback
    elif isinstance(typ, TupleType):
        return typ.partial_fallback
    elif isinstance(typ, TypedDictType):
        return typ.fallback
    elif isinstance(typ, TypeVarType):
        return try_getting_instance_fallback(typ.upper_bound)
    return None

def fixup_partial_type(typ: Type) -> Type:
    if False:
        i = 10
        return i + 15
    "Convert a partial type that we couldn't resolve into something concrete.\n\n    This means, for None we make it Optional[Any], and for anything else we\n    fill in all of the type arguments with Any.\n    "
    if not isinstance(typ, PartialType):
        return typ
    if typ.type is None:
        return UnionType.make_union([AnyType(TypeOfAny.unannotated), NoneType()])
    else:
        return Instance(typ.type, [AnyType(TypeOfAny.unannotated)] * len(typ.type.type_vars))

def get_protocol_member(left: Instance, member: str, class_obj: bool) -> ProperType | None:
    if False:
        for i in range(10):
            print('nop')
    if member == '__call__' and class_obj:
        from mypy.checkmember import type_object_type

        def named_type(fullname: str) -> Instance:
            if False:
                while True:
                    i = 10
            return Instance(left.type.mro[-1], [])
        return type_object_type(left.type, named_type)
    if member == '__call__' and left.type.is_metaclass():
        return None
    from mypy.subtypes import find_member
    return get_proper_type(find_member(member, left, left, class_obj=class_obj))