from __future__ import annotations
from typing import Callable, Sequence
import mypy.subtypes
from mypy.erasetype import erase_typevars
from mypy.expandtype import expand_type
from mypy.nodes import Context
from mypy.types import AnyType, CallableType, ParamSpecType, PartialType, Type, TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType, UninhabitedType, UnpackType, get_proper_type

def get_target_type(tvar: TypeVarLikeType, type: Type, callable: CallableType, report_incompatible_typevar_value: Callable[[CallableType, Type, str, Context], None], context: Context, skip_unsatisfied: bool) -> Type | None:
    if False:
        return 10
    p_type = get_proper_type(type)
    if isinstance(p_type, UninhabitedType) and tvar.has_default():
        return tvar.default
    if isinstance(tvar, ParamSpecType):
        return type
    if isinstance(tvar, TypeVarTupleType):
        return type
    assert isinstance(tvar, TypeVarType)
    values = tvar.values
    if values:
        if isinstance(p_type, AnyType):
            return type
        if isinstance(p_type, TypeVarType) and p_type.values:
            if all((any((mypy.subtypes.is_same_type(v, v1) for v in values)) for v1 in p_type.values)):
                return type
        matching = []
        for value in values:
            if mypy.subtypes.is_subtype(type, value):
                matching.append(value)
        if matching:
            best = matching[0]
            for match in matching[1:]:
                if mypy.subtypes.is_subtype(match, best):
                    best = match
            return best
        if skip_unsatisfied:
            return None
        report_incompatible_typevar_value(callable, type, tvar.name, context)
    else:
        upper_bound = tvar.upper_bound
        if tvar.name == 'Self':
            upper_bound = erase_typevars(upper_bound)
        if not mypy.subtypes.is_subtype(type, upper_bound):
            if skip_unsatisfied:
                return None
            report_incompatible_typevar_value(callable, type, tvar.name, context)
    return type

def apply_generic_arguments(callable: CallableType, orig_types: Sequence[Type | None], report_incompatible_typevar_value: Callable[[CallableType, Type, str, Context], None], context: Context, skip_unsatisfied: bool=False) -> CallableType:
    if False:
        i = 10
        return i + 15
    "Apply generic type arguments to a callable type.\n\n    For example, applying [int] to 'def [T] (T) -> T' results in\n    'def (int) -> int'.\n\n    Note that each type can be None; in this case, it will not be applied.\n\n    If `skip_unsatisfied` is True, then just skip the types that don't satisfy type variable\n    bound or constraints, instead of giving an error.\n    "
    tvars = callable.variables
    assert len(tvars) == len(orig_types)
    id_to_type: dict[TypeVarId, Type] = {}
    for (tvar, type) in zip(tvars, orig_types):
        assert not isinstance(type, PartialType), 'Internal error: must never apply partial type'
        if type is None:
            continue
        target_type = get_target_type(tvar, type, callable, report_incompatible_typevar_value, context, skip_unsatisfied)
        if target_type is not None:
            id_to_type[tvar.id] = target_type
    param_spec = callable.param_spec()
    if param_spec is not None:
        nt = id_to_type.get(param_spec.id)
        if nt is not None:
            callable = expand_type(callable, id_to_type)
            assert isinstance(callable, CallableType)
            return callable.copy_modified(variables=[tv for tv in tvars if tv.id not in id_to_type])
    var_arg = callable.var_arg()
    if var_arg is not None and isinstance(var_arg.typ, UnpackType):
        callable = expand_type(callable, id_to_type)
        assert isinstance(callable, CallableType)
        return callable.copy_modified(variables=[tv for tv in tvars if tv.id not in id_to_type])
    else:
        callable = callable.copy_modified(arg_types=[expand_type(at, id_to_type) for at in callable.arg_types])
    if callable.type_guard is not None:
        type_guard = expand_type(callable.type_guard, id_to_type)
    else:
        type_guard = None
    remaining_tvars = [tv for tv in tvars if tv.id not in id_to_type]
    return callable.copy_modified(ret_type=expand_type(callable.ret_type, id_to_type), variables=remaining_tvars, type_guard=type_guard)