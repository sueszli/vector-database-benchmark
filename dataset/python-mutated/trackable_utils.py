"""Utility methods for the trackable dependencies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections

def pretty_print_node_path(path):
    if False:
        print('Hello World!')
    if not path:
        return 'root object'
    else:
        return 'root.' + '.'.join([p.name for p in path])

class CyclicDependencyError(Exception):

    def __init__(self, leftover_dependency_map):
        if False:
            while True:
                i = 10
        'Creates a CyclicDependencyException.'
        self.leftover_dependency_map = leftover_dependency_map
        super(CyclicDependencyError, self).__init__()

def order_by_dependency(dependency_map):
    if False:
        print('Hello World!')
    "Topologically sorts the keys of a map so that dependencies appear first.\n\n  Uses Kahn's algorithm:\n  https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm\n\n  Args:\n    dependency_map: a dict mapping values to a list of dependencies (other keys\n      in the map). All keys and dependencies must be hashable types.\n\n  Returns:\n    A sorted array of keys from dependency_map.\n\n  Raises:\n    CyclicDependencyError: if there is a cycle in the graph.\n    ValueError: If there are values in the dependency map that are not keys in\n      the map.\n  "
    reverse_dependency_map = collections.defaultdict(set)
    for (x, deps) in dependency_map.items():
        for dep in deps:
            reverse_dependency_map[dep].add(x)
    unknown_keys = reverse_dependency_map.keys() - dependency_map.keys()
    if unknown_keys:
        raise ValueError(f'Found values in the dependency map which are not keys: {unknown_keys}')
    reversed_dependency_arr = []
    to_visit = [x for x in dependency_map if x not in reverse_dependency_map]
    while to_visit:
        x = to_visit.pop(0)
        reversed_dependency_arr.append(x)
        for dep in set(dependency_map[x]):
            edges = reverse_dependency_map[dep]
            edges.remove(x)
            if not edges:
                to_visit.append(dep)
                reverse_dependency_map.pop(dep)
    if reverse_dependency_map:
        leftover_dependency_map = collections.defaultdict(list)
        for (dep, xs) in reverse_dependency_map.items():
            for x in xs:
                leftover_dependency_map[x].append(dep)
        raise CyclicDependencyError(leftover_dependency_map)
    return reversed(reversed_dependency_arr)
_ESCAPE_CHAR = '.'
_OPTIMIZER_SLOTS_NAME = _ESCAPE_CHAR + 'OPTIMIZER_SLOT'
OBJECT_ATTRIBUTES_NAME = _ESCAPE_CHAR + 'ATTRIBUTES'
SERIALIZE_TO_TENSORS_NAME = _ESCAPE_CHAR + 'TENSORS'

def escape_local_name(name):
    if False:
        i = 10
        return i + 15
    return name.replace(_ESCAPE_CHAR, _ESCAPE_CHAR + _ESCAPE_CHAR).replace('/', _ESCAPE_CHAR + 'S')

def object_path_to_string(node_path_arr):
    if False:
        i = 10
        return i + 15
    'Converts a list of nodes to a string.'
    return '/'.join((escape_local_name(trackable.name) for trackable in node_path_arr))

def checkpoint_key(object_path, local_name):
    if False:
        return 10
    'Returns the checkpoint key for a local attribute of an object.'
    key_suffix = escape_local_name(local_name)
    if local_name == SERIALIZE_TO_TENSORS_NAME:
        key_suffix = ''
    return f'{object_path}/{OBJECT_ATTRIBUTES_NAME}/{key_suffix}'

def extract_object_name(key):
    if False:
        print('Hello World!')
    'Substrings the checkpoint key to the start of "/.ATTRIBUTES".'
    search_key = '/' + OBJECT_ATTRIBUTES_NAME
    return key[:key.index(search_key)]

def extract_local_name(key, prefix=None):
    if False:
        print('Hello World!')
    'Returns the substring after the "/.ATTIBUTES/" in the checkpoint key.'
    prefix = prefix or ''
    search_key = OBJECT_ATTRIBUTES_NAME + '/' + prefix
    try:
        return key[key.index(search_key) + len(search_key):]
    except ValueError:
        return key

def slot_variable_key(variable_path, optimizer_path, slot_name):
    if False:
        for i in range(10):
            print('nop')
    'Returns checkpoint key for a slot variable.'
    return f'{variable_path}/{_OPTIMIZER_SLOTS_NAME}/{optimizer_path}/{escape_local_name(slot_name)}'