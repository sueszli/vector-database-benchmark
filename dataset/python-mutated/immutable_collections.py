from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, _register_pytree_node
__all__ = ['immutable_list', 'immutable_dict']
_help_mutation = 'If you are attempting to modify the kwargs or args of a torch.fx.Node object,\ninstead create a new copy of it and assign the copy to the node:\n    new_args = ... # copy and mutate args\n    node.args = new_args\n'

def _no_mutation(self, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    raise NotImplementedError(f"'{type(self).__name__}' object does not support mutation. {_help_mutation}")

def _create_immutable_container(base, mutable_functions):
    if False:
        print('Hello World!')
    container = type('immutable_' + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container
immutable_list = _create_immutable_container(list, ['__delitem__', '__iadd__', '__imul__', '__setitem__', 'append', 'clear', 'extend', 'insert', 'pop', 'remove'])
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))
immutable_list.__hash__ = lambda self: hash(tuple(self))
compatibility(is_backward_compatible=True)(immutable_list)
immutable_dict = _create_immutable_container(dict, ['__delitem__', '__setitem__', 'clear', 'pop', 'popitem', 'update'])
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))
immutable_dict.__hash__ = lambda self: hash(tuple(self.items()))
compatibility(is_backward_compatible=True)(immutable_dict)

def _immutable_dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    if False:
        return 10
    return (list(d.values()), list(d.keys()))

def _immutable_dict_unflatten(values: Iterable[Any], context: Context) -> Dict[Any, Any]:
    if False:
        return 10
    return immutable_dict(dict(zip(context, values)))

def _immutable_list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    if False:
        for i in range(10):
            print('nop')
    return (d, None)

def _immutable_list_unflatten(values: Iterable[Any], context: Context) -> List[Any]:
    if False:
        for i in range(10):
            print('nop')
    return immutable_list(values)
_register_pytree_node(immutable_dict, _immutable_dict_flatten, _immutable_dict_unflatten)
_register_pytree_node(immutable_list, _immutable_list_flatten, _immutable_list_unflatten)