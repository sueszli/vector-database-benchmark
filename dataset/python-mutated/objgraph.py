"""Find all objects reachable from a root object."""
from __future__ import annotations
import types
import weakref
from collections.abc import Iterable
from typing import Final, Iterator, Mapping
method_descriptor_type: Final = type(object.__dir__)
method_wrapper_type: Final = type(object().__ne__)
wrapper_descriptor_type: Final = type(object.__ne__)
FUNCTION_TYPES: Final = (types.BuiltinFunctionType, types.FunctionType, types.MethodType, method_descriptor_type, wrapper_descriptor_type, method_wrapper_type)
ATTR_BLACKLIST: Final = {'__doc__', '__name__', '__class__', '__dict__'}
ATOMIC_TYPE_BLACKLIST: Final = {bool, int, float, str, type(None), object}
COLLECTION_TYPE_BLACKLIST: Final = {list, set, dict, tuple}
TYPE_BLACKLIST: Final = {weakref.ReferenceType}

def isproperty(o: object, attr: str) -> bool:
    if False:
        print('Hello World!')
    return isinstance(getattr(type(o), attr, None), property)

def get_edge_candidates(o: object) -> Iterator[tuple[object, object]]:
    if False:
        return 10
    if '__getattribute__' in getattr(type(o), '__dict__'):
        return
    if type(o) not in COLLECTION_TYPE_BLACKLIST:
        for attr in dir(o):
            try:
                if attr not in ATTR_BLACKLIST and hasattr(o, attr) and (not isproperty(o, attr)):
                    e = getattr(o, attr)
                    if type(e) not in ATOMIC_TYPE_BLACKLIST:
                        yield (attr, e)
            except AssertionError:
                pass
    if isinstance(o, Mapping):
        yield from o.items()
    elif isinstance(o, Iterable) and (not isinstance(o, str)):
        for (i, e) in enumerate(o):
            yield (i, e)

def get_edges(o: object) -> Iterator[tuple[object, object]]:
    if False:
        return 10
    for (s, e) in get_edge_candidates(o):
        if isinstance(e, FUNCTION_TYPES):
            if hasattr(e, '__closure__'):
                yield ((s, '__closure__'), e.__closure__)
            if hasattr(e, '__self__'):
                se = e.__self__
                if se is not o and se is not type(o) and hasattr(s, '__self__'):
                    yield (s.__self__, se)
        elif type(e) not in TYPE_BLACKLIST:
            yield (s, e)

def get_reachable_graph(root: object) -> tuple[dict[int, object], dict[int, tuple[int, object]]]:
    if False:
        return 10
    parents = {}
    seen = {id(root): root}
    worklist = [root]
    while worklist:
        o = worklist.pop()
        for (s, e) in get_edges(o):
            if id(e) in seen:
                continue
            parents[id(e)] = (id(o), s)
            seen[id(e)] = e
            worklist.append(e)
    return (seen, parents)

def get_path(o: object, seen: dict[int, object], parents: dict[int, tuple[int, object]]) -> list[tuple[object, object]]:
    if False:
        return 10
    path = []
    while id(o) in parents:
        (pid, attr) = parents[id(o)]
        o = seen[pid]
        path.append((attr, o))
    path.reverse()
    return path