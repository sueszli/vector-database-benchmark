from __future__ import annotations
from mypy.expandtype import expand_type_by_instance
from mypy.nodes import TypeInfo
from mypy.types import AnyType, Instance, TupleType, TypeOfAny, has_type_vars

def map_instance_to_supertype(instance: Instance, superclass: TypeInfo) -> Instance:
    if False:
        i = 10
        return i + 15
    "Produce a supertype of `instance` that is an Instance\n    of `superclass`, mapping type arguments up the chain of bases.\n\n    If `superclass` is not a nominal superclass of `instance.type`,\n    then all type arguments are mapped to 'Any'.\n    "
    if instance.type == superclass:
        return instance
    if superclass.fullname == 'builtins.tuple' and instance.type.tuple_type:
        if has_type_vars(instance.type.tuple_type):
            alias = instance.type.special_alias
            assert alias is not None
            if not alias._is_recursive:
                tuple_type = expand_type_by_instance(instance.type.tuple_type, instance)
                if isinstance(tuple_type, TupleType):
                    import mypy.typeops
                    return mypy.typeops.tuple_fallback(tuple_type)
                elif isinstance(tuple_type, Instance):
                    return tuple_type
    if not superclass.type_vars:
        return Instance(superclass, [])
    return map_instance_to_supertypes(instance, superclass)[0]

def map_instance_to_supertypes(instance: Instance, supertype: TypeInfo) -> list[Instance]:
    if False:
        while True:
            i = 10
    result: list[Instance] = []
    for path in class_derivation_paths(instance.type, supertype):
        types = [instance]
        for sup in path:
            a: list[Instance] = []
            for t in types:
                a.extend(map_instance_to_direct_supertypes(t, sup))
            types = a
        result.extend(types)
    if result:
        return result
    else:
        any_type = AnyType(TypeOfAny.from_error)
        return [Instance(supertype, [any_type] * len(supertype.type_vars))]

def class_derivation_paths(typ: TypeInfo, supertype: TypeInfo) -> list[list[TypeInfo]]:
    if False:
        i = 10
        return i + 15
    'Return an array of non-empty paths of direct base classes from\n    type to supertype.  Return [] if no such path could be found.\n\n      InterfaceImplementationPaths(A, B) == [[B]] if A inherits B\n      InterfaceImplementationPaths(A, C) == [[B, C]] if A inherits B and\n                                                        B inherits C\n    '
    result: list[list[TypeInfo]] = []
    for base in typ.bases:
        btype = base.type
        if btype == supertype:
            result.append([btype])
        else:
            for path in class_derivation_paths(btype, supertype):
                result.append([btype] + path)
    return result

def map_instance_to_direct_supertypes(instance: Instance, supertype: TypeInfo) -> list[Instance]:
    if False:
        for i in range(10):
            print('nop')
    typ = instance.type
    result: list[Instance] = []
    for b in typ.bases:
        if b.type == supertype:
            t = expand_type_by_instance(b, instance)
            assert isinstance(t, Instance)
            result.append(t)
    if result:
        return result
    else:
        any_type = AnyType(TypeOfAny.unannotated)
        return [Instance(supertype, [any_type] * len(supertype.type_vars))]