from __future__ import annotations
from typing import Callable
from mypy import join
from mypy.erasetype import erase_type
from mypy.maptype import map_instance_to_supertype
from mypy.state import state
from mypy.subtypes import is_callable_compatible, is_equivalent, is_proper_subtype, is_same_type, is_subtype
from mypy.typeops import is_recursive_pair, make_simplified_union, tuple_fallback
from mypy.types import MYPYC_NATIVE_INT_NAMES, TUPLE_LIKE_INSTANCE_NAMES, AnyType, CallableType, DeletedType, ErasedType, FunctionLike, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, ProperType, TupleType, Type, TypeAliasType, TypedDictType, TypeGuardedType, TypeOfAny, TypeType, TypeVarLikeType, TypeVarTupleType, TypeVarType, TypeVisitor, UnboundType, UninhabitedType, UnionType, UnpackType, find_unpack_in_list, get_proper_type, get_proper_types, split_with_prefix_and_suffix

def trivial_meet(s: Type, t: Type) -> ProperType:
    if False:
        while True:
            i = 10
    'Return one of types (expanded) if it is a subtype of other, otherwise bottom type.'
    if is_subtype(s, t):
        return get_proper_type(s)
    elif is_subtype(t, s):
        return get_proper_type(t)
    elif state.strict_optional:
        return UninhabitedType()
    else:
        return NoneType()

def meet_types(s: Type, t: Type) -> ProperType:
    if False:
        for i in range(10):
            print('nop')
    'Return the greatest lower bound of two types.'
    if is_recursive_pair(s, t):
        return trivial_meet(s, t)
    s = get_proper_type(s)
    t = get_proper_type(t)
    if isinstance(s, Instance) and isinstance(t, Instance) and (s.type == t.type):
        if (s.extra_attrs or t.extra_attrs) and is_same_type(s, t):
            if s.extra_attrs and t.extra_attrs:
                if len(s.extra_attrs.attrs) > len(t.extra_attrs.attrs):
                    return s
                return t
            if s.extra_attrs:
                return s
            return t
    if not isinstance(s, UnboundType) and (not isinstance(t, UnboundType)):
        if is_proper_subtype(s, t, ignore_promotions=True):
            return s
        if is_proper_subtype(t, s, ignore_promotions=True):
            return t
    if isinstance(s, ErasedType):
        return s
    if isinstance(s, AnyType):
        return t
    if isinstance(s, UnionType) and (not isinstance(t, UnionType)):
        (s, t) = (t, s)
    (s, t) = join.normalize_callables(s, t)
    return t.accept(TypeMeetVisitor(s))

def narrow_declared_type(declared: Type, narrowed: Type) -> Type:
    if False:
        print('Hello World!')
    'Return the declared type narrowed down to another type.'
    if isinstance(narrowed, TypeGuardedType):
        return narrowed.type_guard
    original_declared = declared
    original_narrowed = narrowed
    declared = get_proper_type(declared)
    narrowed = get_proper_type(narrowed)
    if declared == narrowed:
        return original_declared
    if isinstance(declared, UnionType):
        return make_simplified_union([narrow_declared_type(x, narrowed) for x in declared.relevant_items() if is_overlapping_types(x, narrowed, ignore_promotions=True) or is_subtype(narrowed, x, ignore_promotions=False)])
    if is_enum_overlapping_union(declared, narrowed):
        return original_narrowed
    elif not is_overlapping_types(declared, narrowed, prohibit_none_typevar_overlap=True):
        if state.strict_optional:
            return UninhabitedType()
        else:
            return NoneType()
    elif isinstance(narrowed, UnionType):
        return make_simplified_union([narrow_declared_type(declared, x) for x in narrowed.relevant_items()])
    elif isinstance(narrowed, AnyType):
        return original_narrowed
    elif isinstance(narrowed, TypeVarType) and is_subtype(narrowed.upper_bound, declared):
        return narrowed
    elif isinstance(declared, TypeType) and isinstance(narrowed, TypeType):
        return TypeType.make_normalized(narrow_declared_type(declared.item, narrowed.item))
    elif isinstance(declared, TypeType) and isinstance(narrowed, Instance) and narrowed.type.is_metaclass():
        return original_declared
    elif isinstance(declared, Instance):
        if declared.type.alt_promote:
            return original_declared
        if isinstance(narrowed, Instance) and narrowed.type.alt_promote and (narrowed.type.alt_promote.type is declared.type):
            return original_declared
        return meet_types(original_declared, original_narrowed)
    elif isinstance(declared, (TupleType, TypeType, LiteralType)):
        return meet_types(original_declared, original_narrowed)
    elif isinstance(declared, TypedDictType) and isinstance(narrowed, Instance):
        if narrowed.type.fullname == 'builtins.dict' and all((isinstance(t, AnyType) for t in get_proper_types(narrowed.args))):
            return original_declared
        return meet_types(original_declared, original_narrowed)
    return original_narrowed

def get_possible_variants(typ: Type) -> list[Type]:
    if False:
        i = 10
        return i + 15
    'This function takes any "Union-like" type and returns a list of the available "options".\n\n    Specifically, there are currently exactly three different types that can have\n    "variants" or are "union-like":\n\n    - Unions\n    - TypeVars with value restrictions\n    - Overloads\n\n    This function will return a list of each "option" present in those types.\n\n    If this function receives any other type, we return a list containing just that\n    original type. (E.g. pretend the type was contained within a singleton union).\n\n    The only current exceptions are regular TypeVars and ParamSpecs. For these "TypeVarLike"s,\n    we return a list containing that TypeVarLike\'s upper bound.\n\n    This function is useful primarily when checking to see if two types are overlapping:\n    the algorithm to check if two unions are overlapping is fundamentally the same as\n    the algorithm for checking if two overloads are overlapping.\n\n    Normalizing both kinds of types in the same way lets us reuse the same algorithm\n    for both.\n    '
    typ = get_proper_type(typ)
    if isinstance(typ, TypeVarType):
        if len(typ.values) > 0:
            return typ.values
        else:
            return [typ.upper_bound]
    elif isinstance(typ, ParamSpecType):
        return [typ.upper_bound]
    elif isinstance(typ, TypeVarTupleType):
        return [typ.upper_bound]
    elif isinstance(typ, UnionType):
        return list(typ.items)
    elif isinstance(typ, Overloaded):
        return list(typ.items)
    else:
        return [typ]

def is_enum_overlapping_union(x: ProperType, y: ProperType) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Return True if x is an Enum, and y is an Union with at least one Literal from x'
    return isinstance(x, Instance) and x.type.is_enum and isinstance(y, UnionType) and any((isinstance(p, LiteralType) and x.type == p.fallback.type for p in (get_proper_type(z) for z in y.relevant_items())))

def is_literal_in_union(x: ProperType, y: ProperType) -> bool:
    if False:
        while True:
            i = 10
    'Return True if x is a Literal and y is an Union that includes x'
    return isinstance(x, LiteralType) and isinstance(y, UnionType) and any((x == get_proper_type(z) for z in y.items))

def is_overlapping_types(left: Type, right: Type, ignore_promotions: bool=False, prohibit_none_typevar_overlap: bool=False, ignore_uninhabited: bool=False, seen_types: set[tuple[Type, Type]] | None=None) -> bool:
    if False:
        return 10
    "Can a value of type 'left' also be of type 'right' or vice-versa?\n\n    If 'ignore_promotions' is True, we ignore promotions while checking for overlaps.\n    If 'prohibit_none_typevar_overlap' is True, we disallow None from overlapping with\n    TypeVars (in both strict-optional and non-strict-optional mode).\n    "
    if isinstance(left, TypeGuardedType) or isinstance(right, TypeGuardedType):
        return True
    if seen_types is None:
        seen_types = set()
    if (left, right) in seen_types:
        return True
    if isinstance(left, TypeAliasType) and isinstance(right, TypeAliasType):
        seen_types.add((left, right))
    (left, right) = get_proper_types((left, right))

    def _is_overlapping_types(left: Type, right: Type) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Encode the kind of overlapping check to perform.\n\n        This function mostly exists, so we don't have to repeat keyword arguments everywhere.\n        "
        return is_overlapping_types(left, right, ignore_promotions=ignore_promotions, prohibit_none_typevar_overlap=prohibit_none_typevar_overlap, ignore_uninhabited=ignore_uninhabited, seen_types=seen_types.copy())
    if isinstance(left, PartialType) or isinstance(right, PartialType):
        assert False, 'Unexpectedly encountered partial type'
    illegal_types = (UnboundType, ErasedType, DeletedType)
    if isinstance(left, illegal_types) or isinstance(right, illegal_types):
        return True
    if not state.strict_optional:
        if isinstance(left, UnionType):
            left = UnionType.make_union(left.relevant_items())
        if isinstance(right, UnionType):
            right = UnionType.make_union(right.relevant_items())
        (left, right) = get_proper_types((left, right))
    if isinstance(left, AnyType) or isinstance(right, AnyType):
        return True
    if is_enum_overlapping_union(left, right) or is_enum_overlapping_union(right, left) or is_literal_in_union(left, right) or is_literal_in_union(right, left):
        return True
    if is_proper_subtype(left, right, ignore_promotions=ignore_promotions, ignore_uninhabited=ignore_uninhabited) or is_proper_subtype(right, left, ignore_promotions=ignore_promotions, ignore_uninhabited=ignore_uninhabited):
        return True
    left_possible = get_possible_variants(left)
    right_possible = get_possible_variants(right)
    if isinstance(left, (Parameters, ParamSpecType)) and isinstance(right, (Parameters, ParamSpecType)):
        return True
    if isinstance(left, Parameters) or isinstance(right, Parameters):
        return False

    def is_none_typevarlike_overlap(t1: Type, t2: Type) -> bool:
        if False:
            while True:
                i = 10
        (t1, t2) = get_proper_types((t1, t2))
        return isinstance(t1, NoneType) and isinstance(t2, TypeVarLikeType)
    if prohibit_none_typevar_overlap:
        if is_none_typevarlike_overlap(left, right) or is_none_typevarlike_overlap(right, left):
            return False
    if len(left_possible) > 1 or len(right_possible) > 1 or isinstance(left, TypeVarLikeType) or isinstance(right, TypeVarLikeType):
        for l in left_possible:
            for r in right_possible:
                if _is_overlapping_types(l, r):
                    return True
        return False
    if state.strict_optional and isinstance(left, NoneType) != isinstance(right, NoneType):
        return False
    if isinstance(left, TypedDictType) and isinstance(right, TypedDictType):
        return are_typed_dicts_overlapping(left, right, ignore_promotions=ignore_promotions)
    elif typed_dict_mapping_pair(left, right):
        return typed_dict_mapping_overlap(left, right, overlapping=_is_overlapping_types)
    elif isinstance(left, TypedDictType):
        left = left.fallback
    elif isinstance(right, TypedDictType):
        right = right.fallback
    if is_tuple(left) and is_tuple(right):
        return are_tuples_overlapping(left, right, ignore_promotions=ignore_promotions)
    elif isinstance(left, TupleType):
        left = tuple_fallback(left)
    elif isinstance(right, TupleType):
        right = tuple_fallback(right)
    if isinstance(left, TypeType) and isinstance(right, TypeType):
        return _is_overlapping_types(left.item, right.item)

    def _type_object_overlap(left: Type, right: Type) -> bool:
        if False:
            while True:
                i = 10
        'Special cases for type object types overlaps.'
        (left, right) = get_proper_types((left, right))
        if isinstance(left, TypeType) and isinstance(right, CallableType):
            return _is_overlapping_types(left.item, right.ret_type)
        if isinstance(left, TypeType) and isinstance(right, Instance):
            if isinstance(left.item, Instance):
                left_meta = left.item.type.metaclass_type
                if left_meta is not None:
                    return _is_overlapping_types(left_meta, right)
                return right.type.has_base('builtins.type')
            elif isinstance(left.item, AnyType):
                return right.type.has_base('builtins.type')
        return False
    if isinstance(left, TypeType) or isinstance(right, TypeType):
        return _type_object_overlap(left, right) or _type_object_overlap(right, left)
    if isinstance(left, CallableType) and isinstance(right, CallableType):
        return is_callable_compatible(left, right, is_compat=_is_overlapping_types, is_proper_subtype=False, ignore_pos_arg_names=True, allow_partial_overlap=True)
    elif isinstance(left, CallableType):
        left = left.fallback
    elif isinstance(right, CallableType):
        right = right.fallback
    if isinstance(left, LiteralType) and isinstance(right, LiteralType):
        if left.value == right.value:
            left = left.fallback
            right = right.fallback
        else:
            return False
    elif isinstance(left, LiteralType):
        left = left.fallback
    elif isinstance(right, LiteralType):
        right = right.fallback
    if isinstance(left, Instance) and isinstance(right, Instance):
        if is_subtype(left, right, ignore_promotions=ignore_promotions, ignore_uninhabited=ignore_uninhabited) or is_subtype(right, left, ignore_promotions=ignore_promotions, ignore_uninhabited=ignore_uninhabited):
            return True
        if right.type.fullname == 'builtins.int' and left.type.fullname in MYPYC_NATIVE_INT_NAMES:
            return True
        if left.type.has_base(right.type.fullname):
            left = map_instance_to_supertype(left, right.type)
        elif right.type.has_base(left.type.fullname):
            right = map_instance_to_supertype(right, left.type)
        else:
            return False
        if len(left.args) == len(right.args):
            if all((_is_overlapping_types(left_arg, right_arg) for (left_arg, right_arg) in zip(left.args, right.args))):
                return True
        return False
    assert type(left) != type(right), f'{type(left)} vs {type(right)}'
    return False

def is_overlapping_erased_types(left: Type, right: Type, *, ignore_promotions: bool=False) -> bool:
    if False:
        print('Hello World!')
    "The same as 'is_overlapping_erased_types', except the types are erased first."
    return is_overlapping_types(erase_type(left), erase_type(right), ignore_promotions=ignore_promotions, prohibit_none_typevar_overlap=True)

def are_typed_dicts_overlapping(left: TypedDictType, right: TypedDictType, *, ignore_promotions: bool=False, prohibit_none_typevar_overlap: bool=False) -> bool:
    if False:
        return 10
    "Returns 'true' if left and right are overlapping TypeDictTypes."
    for key in left.required_keys:
        if key not in right.items:
            return False
        if not is_overlapping_types(left.items[key], right.items[key], ignore_promotions=ignore_promotions, prohibit_none_typevar_overlap=prohibit_none_typevar_overlap):
            return False
    for key in right.required_keys:
        if key not in left.items:
            return False
        if not is_overlapping_types(left.items[key], right.items[key], ignore_promotions=ignore_promotions):
            return False
    return True

def are_tuples_overlapping(left: Type, right: Type, *, ignore_promotions: bool=False, prohibit_none_typevar_overlap: bool=False) -> bool:
    if False:
        return 10
    'Returns true if left and right are overlapping tuples.'
    (left, right) = get_proper_types((left, right))
    left = adjust_tuple(left, right) or left
    right = adjust_tuple(right, left) or right
    assert isinstance(left, TupleType), f'Type {left} is not a tuple'
    assert isinstance(right, TupleType), f'Type {right} is not a tuple'
    if len(left.items) != len(right.items):
        return False
    return all((is_overlapping_types(l, r, ignore_promotions=ignore_promotions, prohibit_none_typevar_overlap=prohibit_none_typevar_overlap) for (l, r) in zip(left.items, right.items)))

def adjust_tuple(left: ProperType, r: ProperType) -> TupleType | None:
    if False:
        for i in range(10):
            print('nop')
    'Find out if `left` is a Tuple[A, ...], and adjust its length to `right`'
    if isinstance(left, Instance) and left.type.fullname == 'builtins.tuple':
        n = r.length() if isinstance(r, TupleType) else 1
        return TupleType([left.args[0]] * n, left)
    return None

def is_tuple(typ: Type) -> bool:
    if False:
        i = 10
        return i + 15
    typ = get_proper_type(typ)
    return isinstance(typ, TupleType) or (isinstance(typ, Instance) and typ.type.fullname == 'builtins.tuple')

class TypeMeetVisitor(TypeVisitor[ProperType]):

    def __init__(self, s: ProperType) -> None:
        if False:
            return 10
        self.s = s

    def visit_unbound_type(self, t: UnboundType) -> ProperType:
        if False:
            return 10
        if isinstance(self.s, NoneType):
            if state.strict_optional:
                return AnyType(TypeOfAny.special_form)
            else:
                return self.s
        elif isinstance(self.s, UninhabitedType):
            return self.s
        else:
            return AnyType(TypeOfAny.special_form)

    def visit_any(self, t: AnyType) -> ProperType:
        if False:
            return 10
        return self.s

    def visit_union_type(self, t: UnionType) -> ProperType:
        if False:
            i = 10
            return i + 15
        if isinstance(self.s, UnionType):
            meets: list[Type] = []
            for x in t.items:
                for y in self.s.items:
                    meets.append(meet_types(x, y))
        else:
            meets = [meet_types(x, self.s) for x in t.items]
        return make_simplified_union(meets)

    def visit_none_type(self, t: NoneType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        if state.strict_optional:
            if isinstance(self.s, NoneType) or (isinstance(self.s, Instance) and self.s.type.fullname == 'builtins.object'):
                return t
            else:
                return UninhabitedType()
        else:
            return t

    def visit_uninhabited_type(self, t: UninhabitedType) -> ProperType:
        if False:
            while True:
                i = 10
        return t

    def visit_deleted_type(self, t: DeletedType) -> ProperType:
        if False:
            print('Hello World!')
        if isinstance(self.s, NoneType):
            if state.strict_optional:
                return t
            else:
                return self.s
        elif isinstance(self.s, UninhabitedType):
            return self.s
        else:
            return t

    def visit_erased_type(self, t: ErasedType) -> ProperType:
        if False:
            return 10
        return self.s

    def visit_type_var(self, t: TypeVarType) -> ProperType:
        if False:
            while True:
                i = 10
        if isinstance(self.s, TypeVarType) and self.s.id == t.id:
            return self.s
        else:
            return self.default(self.s)

    def visit_param_spec(self, t: ParamSpecType) -> ProperType:
        if False:
            print('Hello World!')
        if self.s == t:
            return self.s
        else:
            return self.default(self.s)

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> ProperType:
        if False:
            print('Hello World!')
        if isinstance(self.s, TypeVarTupleType) and self.s.id == t.id:
            return self.s if self.s.min_len > t.min_len else t
        else:
            return self.default(self.s)

    def visit_unpack_type(self, t: UnpackType) -> ProperType:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def visit_parameters(self, t: Parameters) -> ProperType:
        if False:
            while True:
                i = 10
        if isinstance(self.s, Parameters):
            if len(t.arg_types) != len(self.s.arg_types):
                return self.default(self.s)
            from mypy.join import join_types
            return t.copy_modified(arg_types=[join_types(s_a, t_a) for (s_a, t_a) in zip(self.s.arg_types, t.arg_types)])
        else:
            return self.default(self.s)

    def visit_instance(self, t: Instance) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.s, Instance):
            if t.type == self.s.type:
                if is_subtype(t, self.s) or is_subtype(self.s, t):
                    args: list[Type] = []
                    if t.type.has_type_var_tuple_type:
                        s = self.s
                        assert s.type.type_var_tuple_prefix is not None
                        assert s.type.type_var_tuple_suffix is not None
                        prefix = s.type.type_var_tuple_prefix
                        suffix = s.type.type_var_tuple_suffix
                        tvt = s.type.defn.type_vars[prefix]
                        assert isinstance(tvt, TypeVarTupleType)
                        fallback = tvt.tuple_fallback
                        (s_prefix, s_middle, s_suffix) = split_with_prefix_and_suffix(s.args, prefix, suffix)
                        (t_prefix, t_middle, t_suffix) = split_with_prefix_and_suffix(t.args, prefix, suffix)
                        s_args = s_prefix + (TupleType(list(s_middle), fallback),) + s_suffix
                        t_args = t_prefix + (TupleType(list(t_middle), fallback),) + t_suffix
                    else:
                        t_args = t.args
                        s_args = self.s.args
                    for (ta, sa, tv) in zip(t_args, s_args, t.type.defn.type_vars):
                        meet = self.meet(ta, sa)
                        if isinstance(tv, TypeVarTupleType):
                            if isinstance(meet, TupleType):
                                args.extend(meet.items)
                                continue
                            else:
                                assert isinstance(meet, UninhabitedType)
                                meet = UnpackType(tv.tuple_fallback.copy_modified(args=[meet]))
                        args.append(meet)
                    return Instance(t.type, args)
                elif state.strict_optional:
                    return UninhabitedType()
                else:
                    return NoneType()
            else:
                alt_promote = t.type.alt_promote
                if alt_promote and alt_promote.type is self.s.type:
                    return t
                alt_promote = self.s.type.alt_promote
                if alt_promote and alt_promote.type is t.type:
                    return self.s
                if is_subtype(t, self.s):
                    return t
                elif is_subtype(self.s, t):
                    return self.s
                elif state.strict_optional:
                    return UninhabitedType()
                else:
                    return NoneType()
        elif isinstance(self.s, FunctionLike) and t.type.is_protocol:
            call = join.unpack_callback_protocol(t)
            if call:
                return meet_types(call, self.s)
        elif isinstance(self.s, FunctionLike) and self.s.is_type_obj() and t.type.is_metaclass():
            if is_subtype(self.s.fallback, t):
                return self.s
            return self.default(self.s)
        elif isinstance(self.s, TypeType):
            return meet_types(t, self.s)
        elif isinstance(self.s, TupleType):
            return meet_types(t, self.s)
        elif isinstance(self.s, LiteralType):
            return meet_types(t, self.s)
        elif isinstance(self.s, TypedDictType):
            return meet_types(t, self.s)
        return self.default(self.s)

    def visit_callable_type(self, t: CallableType) -> ProperType:
        if False:
            return 10
        if isinstance(self.s, CallableType) and join.is_similar_callables(t, self.s):
            if is_equivalent(t, self.s):
                return join.combine_similar_callables(t, self.s)
            result = meet_similar_callables(t, self.s)
            if not (t.is_type_obj() and t.type_object().is_abstract or (self.s.is_type_obj() and self.s.type_object().is_abstract)):
                result.from_type_type = True
            if isinstance(get_proper_type(result.ret_type), UninhabitedType):
                return self.default(self.s)
            return result
        elif isinstance(self.s, TypeType) and t.is_type_obj() and (not t.is_generic()):
            res = meet_types(self.s.item, t.ret_type)
            if not isinstance(res, (NoneType, UninhabitedType)):
                return TypeType.make_normalized(res)
            return self.default(self.s)
        elif isinstance(self.s, Instance) and self.s.type.is_protocol:
            call = join.unpack_callback_protocol(self.s)
            if call:
                return meet_types(t, call)
        return self.default(self.s)

    def visit_overloaded(self, t: Overloaded) -> ProperType:
        if False:
            while True:
                i = 10
        s = self.s
        if isinstance(s, FunctionLike):
            if s.items == t.items:
                return Overloaded(t.items)
            elif is_subtype(s, t):
                return s
            elif is_subtype(t, s):
                return t
            else:
                return meet_types(t.fallback, s.fallback)
        elif isinstance(self.s, Instance) and self.s.type.is_protocol:
            call = join.unpack_callback_protocol(self.s)
            if call:
                return meet_types(t, call)
        return meet_types(t.fallback, s)

    def meet_tuples(self, s: TupleType, t: TupleType) -> list[Type] | None:
        if False:
            while True:
                i = 10
        "Meet two tuple types while handling variadic entries.\n\n        This is surprisingly tricky, and we don't handle some tricky corner cases.\n        Most of the trickiness comes from the variadic tuple items like *tuple[X, ...]\n        since they can have arbitrary partial overlaps (while *Ts can't be split). This\n        function is roughly a mirror of join_tuples() w.r.t. to the fact that fixed\n        tuples are subtypes of variadic ones but not vice versa.\n        "
        s_unpack_index = find_unpack_in_list(s.items)
        t_unpack_index = find_unpack_in_list(t.items)
        if s_unpack_index is None and t_unpack_index is None:
            if s.length() == t.length():
                items: list[Type] = []
                for i in range(t.length()):
                    items.append(self.meet(t.items[i], s.items[i]))
                return items
            return None
        if s_unpack_index is not None and t_unpack_index is not None:
            if s.length() == t.length() and s_unpack_index == t_unpack_index:
                unpack_index = s_unpack_index
                s_unpack = s.items[unpack_index]
                assert isinstance(s_unpack, UnpackType)
                s_unpacked = get_proper_type(s_unpack.type)
                t_unpack = t.items[unpack_index]
                assert isinstance(t_unpack, UnpackType)
                t_unpacked = get_proper_type(t_unpack.type)
                if not (isinstance(s_unpacked, Instance) and isinstance(t_unpacked, Instance)):
                    return None
                meet = self.meet(s_unpacked, t_unpacked)
                if not isinstance(meet, Instance):
                    return None
                m_prefix: list[Type] = []
                for (si, ti) in zip(s.items[:unpack_index], t.items[:unpack_index]):
                    m_prefix.append(meet_types(si, ti))
                m_suffix: list[Type] = []
                for (si, ti) in zip(s.items[unpack_index + 1:], t.items[unpack_index + 1:]):
                    m_suffix.append(meet_types(si, ti))
                return m_prefix + [UnpackType(meet)] + m_suffix
            return None
        if s_unpack_index is not None:
            variadic = s
            unpack_index = s_unpack_index
            fixed = t
        else:
            assert t_unpack_index is not None
            variadic = t
            unpack_index = t_unpack_index
            fixed = s
        unpack = variadic.items[unpack_index]
        assert isinstance(unpack, UnpackType)
        unpacked = get_proper_type(unpack.type)
        if not isinstance(unpacked, Instance):
            return None
        if fixed.length() < variadic.length() - 1:
            return None
        prefix_len = unpack_index
        suffix_len = variadic.length() - prefix_len - 1
        (prefix, middle, suffix) = split_with_prefix_and_suffix(tuple(fixed.items), prefix_len, suffix_len)
        items = []
        for (fi, vi) in zip(prefix, variadic.items[:prefix_len]):
            items.append(self.meet(fi, vi))
        for mi in middle:
            items.append(self.meet(mi, unpacked.args[0]))
        if suffix_len:
            for (fi, vi) in zip(suffix, variadic.items[-suffix_len:]):
                items.append(self.meet(fi, vi))
        return items

    def visit_tuple_type(self, t: TupleType) -> ProperType:
        if False:
            i = 10
            return i + 15
        if isinstance(self.s, TupleType):
            items = self.meet_tuples(self.s, t)
            if items is None:
                return self.default(self.s)
            return TupleType(items, tuple_fallback(t))
        elif isinstance(self.s, Instance):
            if self.s.type.fullname in TUPLE_LIKE_INSTANCE_NAMES and self.s.args:
                return t.copy_modified(items=[meet_types(it, self.s.args[0]) for it in t.items])
            elif is_proper_subtype(t, self.s):
                return t
            elif self.s.type.has_type_var_tuple_type and is_subtype(t, self.s):
                return t
        return self.default(self.s)

    def visit_typeddict_type(self, t: TypedDictType) -> ProperType:
        if False:
            print('Hello World!')
        if isinstance(self.s, TypedDictType):
            for (name, l, r) in self.s.zip(t):
                if not is_equivalent(l, r) or (name in t.required_keys) != (name in self.s.required_keys):
                    return self.default(self.s)
            item_list: list[tuple[str, Type]] = []
            for (item_name, s_item_type, t_item_type) in self.s.zipall(t):
                if s_item_type is not None:
                    item_list.append((item_name, s_item_type))
                else:
                    assert t_item_type is not None
                    item_list.append((item_name, t_item_type))
            items = dict(item_list)
            fallback = self.s.create_anonymous_fallback()
            required_keys = t.required_keys | self.s.required_keys
            return TypedDictType(items, required_keys, fallback)
        elif isinstance(self.s, Instance) and is_subtype(t, self.s):
            return t
        else:
            return self.default(self.s)

    def visit_literal_type(self, t: LiteralType) -> ProperType:
        if False:
            print('Hello World!')
        if isinstance(self.s, LiteralType) and self.s == t:
            return t
        elif isinstance(self.s, Instance) and is_subtype(t.fallback, self.s):
            return t
        else:
            return self.default(self.s)

    def visit_partial_type(self, t: PartialType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        assert False, 'Internal error'

    def visit_type_type(self, t: TypeType) -> ProperType:
        if False:
            print('Hello World!')
        if isinstance(self.s, TypeType):
            typ = self.meet(t.item, self.s.item)
            if not isinstance(typ, NoneType):
                typ = TypeType.make_normalized(typ, line=t.line)
            return typ
        elif isinstance(self.s, Instance) and self.s.type.fullname == 'builtins.type':
            return t
        elif isinstance(self.s, CallableType):
            return self.meet(t, self.s)
        else:
            return self.default(self.s)

    def visit_type_alias_type(self, t: TypeAliasType) -> ProperType:
        if False:
            while True:
                i = 10
        assert False, f'This should be never called, got {t}'

    def meet(self, s: Type, t: Type) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return meet_types(s, t)

    def default(self, typ: Type) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(typ, UnboundType):
            return AnyType(TypeOfAny.special_form)
        elif state.strict_optional:
            return UninhabitedType()
        else:
            return NoneType()

def meet_similar_callables(t: CallableType, s: CallableType) -> CallableType:
    if False:
        for i in range(10):
            print('nop')
    from mypy.join import safe_join
    arg_types: list[Type] = []
    for i in range(len(t.arg_types)):
        arg_types.append(safe_join(t.arg_types[i], s.arg_types[i]))
    if t.fallback.type.fullname != 'builtins.function':
        fallback = t.fallback
    else:
        fallback = s.fallback
    return t.copy_modified(arg_types=arg_types, ret_type=meet_types(t.ret_type, s.ret_type), fallback=fallback, name=None)

def meet_type_list(types: list[Type]) -> Type:
    if False:
        while True:
            i = 10
    if not types:
        return AnyType(TypeOfAny.implementation_artifact)
    met = types[0]
    for t in types[1:]:
        met = meet_types(met, t)
    return met

def typed_dict_mapping_pair(left: Type, right: Type) -> bool:
    if False:
        while True:
            i = 10
    'Is this a pair where one type is a TypedDict and another one is an instance of Mapping?\n\n    This case requires a precise/principled consideration because there are two use cases\n    that push the boundary the opposite ways: we need to avoid spurious overlaps to avoid\n    false positives for overloads, but we also need to avoid spuriously non-overlapping types\n    to avoid false positives with --strict-equality.\n    '
    (left, right) = get_proper_types((left, right))
    assert not isinstance(left, TypedDictType) or not isinstance(right, TypedDictType)
    if isinstance(left, TypedDictType):
        (_, other) = (left, right)
    elif isinstance(right, TypedDictType):
        (_, other) = (right, left)
    else:
        return False
    return isinstance(other, Instance) and other.type.has_base('typing.Mapping')

def typed_dict_mapping_overlap(left: Type, right: Type, overlapping: Callable[[Type, Type], bool]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Check if a TypedDict type is overlapping with a Mapping.\n\n    The basic logic here consists of two rules:\n\n    * A TypedDict with some required keys is overlapping with Mapping[str, <some type>]\n      if and only if every key type is overlapping with <some type>. For example:\n\n      - TypedDict(x=int, y=str) overlaps with Dict[str, Union[str, int]]\n      - TypedDict(x=int, y=str) doesn't overlap with Dict[str, int]\n\n      Note that any additional non-required keys can't change the above result.\n\n    * A TypedDict with no required keys overlaps with Mapping[str, <some type>] if and\n      only if at least one of key types overlaps with <some type>. For example:\n\n      - TypedDict(x=str, y=str, total=False) overlaps with Dict[str, str]\n      - TypedDict(x=str, y=str, total=False) doesn't overlap with Dict[str, int]\n      - TypedDict(x=int, y=str, total=False) overlaps with Dict[str, str]\n\n    As usual empty, dictionaries lie in a gray area. In general, List[str] and List[str]\n    are considered non-overlapping despite empty list belongs to both. However, List[int]\n    and List[Never] are considered overlapping.\n\n    So here we follow the same logic: a TypedDict with no required keys is considered\n    non-overlapping with Mapping[str, <some type>], but is considered overlapping with\n    Mapping[Never, Never]. This way we avoid false positives for overloads, and also\n    avoid false positives for comparisons like SomeTypedDict == {} under --strict-equality.\n    "
    (left, right) = get_proper_types((left, right))
    assert not isinstance(left, TypedDictType) or not isinstance(right, TypedDictType)
    if isinstance(left, TypedDictType):
        assert isinstance(right, Instance)
        (typed, other) = (left, right)
    else:
        assert isinstance(left, Instance)
        assert isinstance(right, TypedDictType)
        (typed, other) = (right, left)
    mapping = next((base for base in other.type.mro if base.fullname == 'typing.Mapping'))
    other = map_instance_to_supertype(other, mapping)
    (key_type, value_type) = get_proper_types(other.args)
    fallback = typed.as_anonymous().fallback
    str_type = fallback.type.bases[0].args[0]
    if isinstance(key_type, UninhabitedType) and isinstance(value_type, UninhabitedType):
        return not typed.required_keys
    if typed.required_keys:
        if not overlapping(key_type, str_type):
            return False
        return all((overlapping(typed.items[k], value_type) for k in typed.required_keys))
    else:
        if not overlapping(key_type, str_type):
            return False
        non_required = set(typed.items.keys()) - typed.required_keys
        return any((overlapping(typed.items[k], value_type) for k in non_required))