from __future__ import annotations
from typing import Callable, Container, cast
from mypy.nodes import ARG_STAR, ARG_STAR2
from mypy.types import AnyType, CallableType, DeletedType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, ProperType, TupleType, Type, TypeAliasType, TypedDictType, TypeOfAny, TypeTranslator, TypeType, TypeVarId, TypeVarTupleType, TypeVarType, TypeVisitor, UnboundType, UninhabitedType, UnionType, UnpackType, get_proper_type, get_proper_types

def erase_type(typ: Type) -> ProperType:
    if False:
        return 10
    'Erase any type variables from a type.\n\n    Also replace tuple types with the corresponding concrete types.\n\n    Examples:\n      A -> A\n      B[X] -> B[Any]\n      Tuple[A, B] -> tuple\n      Callable[[A1, A2, ...], R] -> Callable[..., Any]\n      Type[X] -> Type[Any]\n    '
    typ = get_proper_type(typ)
    return typ.accept(EraseTypeVisitor())

class EraseTypeVisitor(TypeVisitor[ProperType]):

    def visit_unbound_type(self, t: UnboundType) -> ProperType:
        if False:
            while True:
                i = 10
        return AnyType(TypeOfAny.from_error)

    def visit_any(self, t: AnyType) -> ProperType:
        if False:
            return 10
        return t

    def visit_none_type(self, t: NoneType) -> ProperType:
        if False:
            i = 10
            return i + 15
        return t

    def visit_uninhabited_type(self, t: UninhabitedType) -> ProperType:
        if False:
            print('Hello World!')
        return t

    def visit_erased_type(self, t: ErasedType) -> ProperType:
        if False:
            print('Hello World!')
        return t

    def visit_partial_type(self, t: PartialType) -> ProperType:
        if False:
            return 10
        raise RuntimeError('Cannot erase partial types')

    def visit_deleted_type(self, t: DeletedType) -> ProperType:
        if False:
            print('Hello World!')
        return t

    def visit_instance(self, t: Instance) -> ProperType:
        if False:
            return 10
        args: list[Type] = []
        for tv in t.type.defn.type_vars:
            if isinstance(tv, TypeVarTupleType):
                args.append(UnpackType(tv.tuple_fallback.copy_modified(args=[AnyType(TypeOfAny.special_form)])))
            else:
                args.append(AnyType(TypeOfAny.special_form))
        return Instance(t.type, args, t.line)

    def visit_type_var(self, t: TypeVarType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return AnyType(TypeOfAny.special_form)

    def visit_param_spec(self, t: ParamSpecType) -> ProperType:
        if False:
            print('Hello World!')
        return AnyType(TypeOfAny.special_form)

    def visit_parameters(self, t: Parameters) -> ProperType:
        if False:
            return 10
        raise RuntimeError('Parameters should have been bound to a class')

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return t.tuple_fallback.copy_modified(args=[AnyType(TypeOfAny.special_form)])

    def visit_unpack_type(self, t: UnpackType) -> ProperType:
        if False:
            for i in range(10):
                print('nop')
        return AnyType(TypeOfAny.special_form)

    def visit_callable_type(self, t: CallableType) -> ProperType:
        if False:
            i = 10
            return i + 15
        any_type = AnyType(TypeOfAny.special_form)
        return CallableType(arg_types=[any_type, any_type], arg_kinds=[ARG_STAR, ARG_STAR2], arg_names=[None, None], ret_type=any_type, fallback=t.fallback, is_ellipsis_args=True, implicit=True)

    def visit_overloaded(self, t: Overloaded) -> ProperType:
        if False:
            return 10
        return t.fallback.accept(self)

    def visit_tuple_type(self, t: TupleType) -> ProperType:
        if False:
            print('Hello World!')
        return t.partial_fallback.accept(self)

    def visit_typeddict_type(self, t: TypedDictType) -> ProperType:
        if False:
            return 10
        return t.fallback.accept(self)

    def visit_literal_type(self, t: LiteralType) -> ProperType:
        if False:
            while True:
                i = 10
        return t

    def visit_union_type(self, t: UnionType) -> ProperType:
        if False:
            i = 10
            return i + 15
        erased_items = [erase_type(item) for item in t.items]
        from mypy.typeops import make_simplified_union
        return make_simplified_union(erased_items)

    def visit_type_type(self, t: TypeType) -> ProperType:
        if False:
            print('Hello World!')
        return TypeType.make_normalized(t.item.accept(self), line=t.line)

    def visit_type_alias_type(self, t: TypeAliasType) -> ProperType:
        if False:
            while True:
                i = 10
        raise RuntimeError('Type aliases should be expanded before accepting this visitor')

def erase_typevars(t: Type, ids_to_erase: Container[TypeVarId] | None=None) -> Type:
    if False:
        return 10
    'Replace all type variables in a type with any,\n    or just the ones in the provided collection.\n    '

    def erase_id(id: TypeVarId) -> bool:
        if False:
            i = 10
            return i + 15
        if ids_to_erase is None:
            return True
        return id in ids_to_erase
    return t.accept(TypeVarEraser(erase_id, AnyType(TypeOfAny.special_form)))

def replace_meta_vars(t: Type, target_type: Type) -> Type:
    if False:
        while True:
            i = 10
    'Replace unification variables in a type with the target type.'
    return t.accept(TypeVarEraser(lambda id: id.is_meta_var(), target_type))

class TypeVarEraser(TypeTranslator):
    """Implementation of type erasure"""

    def __init__(self, erase_id: Callable[[TypeVarId], bool], replacement: Type) -> None:
        if False:
            return 10
        self.erase_id = erase_id
        self.replacement = replacement

    def visit_type_var(self, t: TypeVarType) -> Type:
        if False:
            return 10
        if self.erase_id(t.id):
            return self.replacement
        return t

    def visit_instance(self, t: Instance) -> Type:
        if False:
            for i in range(10):
                print('nop')
        result = super().visit_instance(t)
        assert isinstance(result, ProperType) and isinstance(result, Instance)
        if t.type.fullname == 'builtins.tuple':
            arg = result.args[0]
            if isinstance(arg, UnpackType):
                unpacked = get_proper_type(arg.type)
                if isinstance(unpacked, Instance):
                    assert unpacked.type.fullname == 'builtins.tuple'
                    return unpacked
        return result

    def visit_tuple_type(self, t: TupleType) -> Type:
        if False:
            print('Hello World!')
        result = super().visit_tuple_type(t)
        assert isinstance(result, ProperType) and isinstance(result, TupleType)
        if len(result.items) == 1:
            item = result.items[0]
            if isinstance(item, UnpackType):
                unpacked = get_proper_type(item.type)
                if isinstance(unpacked, Instance):
                    assert unpacked.type.fullname == 'builtins.tuple'
                    if result.partial_fallback.type.fullname != 'builtins.tuple':
                        return result.partial_fallback.accept(self)
                    return unpacked
        return result

    def visit_type_var_tuple(self, t: TypeVarTupleType) -> Type:
        if False:
            return 10
        if self.erase_id(t.id):
            return t.tuple_fallback.copy_modified(args=[self.replacement])
        return t

    def visit_param_spec(self, t: ParamSpecType) -> Type:
        if False:
            i = 10
            return i + 15
        if self.erase_id(t.id):
            return self.replacement
        return t

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            return 10
        return t.copy_modified(args=[a.accept(self) for a in t.args])

def remove_instance_last_known_values(t: Type) -> Type:
    if False:
        while True:
            i = 10
    return t.accept(LastKnownValueEraser())

class LastKnownValueEraser(TypeTranslator):
    """Removes the Literal[...] type that may be associated with any
    Instance types."""

    def visit_instance(self, t: Instance) -> Type:
        if False:
            print('Hello World!')
        if not t.last_known_value and (not t.args):
            return t
        return t.copy_modified(args=[a.accept(self) for a in t.args], last_known_value=None)

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:
        if False:
            i = 10
            return i + 15
        return t

    def visit_union_type(self, t: UnionType) -> Type:
        if False:
            while True:
                i = 10
        new = cast(UnionType, super().visit_union_type(t))
        instances = [item for item in new.items if isinstance(get_proper_type(item), Instance)]
        if len(instances) > 1:
            instances_by_name: dict[str, list[Instance]] = {}
            p_new_items = get_proper_types(new.items)
            for p_item in p_new_items:
                if isinstance(p_item, Instance) and (not p_item.args):
                    instances_by_name.setdefault(p_item.type.fullname, []).append(p_item)
            merged: list[Type] = []
            for item in new.items:
                orig_item = item
                item = get_proper_type(item)
                if isinstance(item, Instance) and (not item.args):
                    types = instances_by_name.get(item.type.fullname)
                    if types is not None:
                        if len(types) == 1:
                            merged.append(item)
                        else:
                            from mypy.typeops import make_simplified_union
                            merged.append(make_simplified_union(types))
                            del instances_by_name[item.type.fullname]
                else:
                    merged.append(orig_item)
            return UnionType.make_union(merged)
        return new