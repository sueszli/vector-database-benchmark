from typing import TYPE_CHECKING, Set, Iterable
if TYPE_CHECKING:
    from .. import Resource
_DEPENDENCIES_PROPERTY = '_direct_computed_dependencies'

def declare_dependency(from_resource: 'Resource', to_resource: 'Resource') -> bool:
    if False:
        print('Hello World!')
    'Remembers that `from_resource` depends on `to_resource`, unless\n    adding this dependency would form a cycle to the known\n    dependency graph. Returns `True` if successful, `False` if\n    skipped due to cycles.\n\n    '
    if _reachable(from_resource=to_resource, to_resource=from_resource):
        return False
    _add_dep(from_resource, to_resource)
    return True

def _deps(res: 'Resource') -> Set['Resource']:
    if False:
        for i in range(10):
            print('nop')
    return getattr(res, _DEPENDENCIES_PROPERTY, set())

def _add_dep(from_resource: 'Resource', to_resource: 'Resource') -> None:
    if False:
        print('Hello World!')
    return setattr(from_resource, _DEPENDENCIES_PROPERTY, _deps(from_resource) | set([to_resource]))

def _reachable(from_resource: 'Resource', to_resource: 'Resource') -> bool:
    if False:
        while True:
            i = 10
    visited: Set['Resource'] = set()
    for x in _with_transitive_deps(from_resource, visited):
        if x == to_resource:
            return True
    return False

def _with_transitive_deps(r: 'Resource', visited: Set['Resource']) -> Iterable['Resource']:
    if False:
        while True:
            i = 10
    if r in visited:
        return
    visited.add(r)
    yield r
    for x in _deps(r):
        for y in _with_transitive_deps(x, visited):
            yield y