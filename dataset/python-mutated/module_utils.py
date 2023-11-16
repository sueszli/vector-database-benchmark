import contextlib
from ..module import Sequential
from ..module.module import Module, _access_structure
from ..tensor import Tensor

def get_expand_structure(obj: Module, key: str):
    if False:
        print('Hello World!')
    "Gets Module's attribute compatible with complex key from Module's :meth:`~.named_children`.\n    Supports handling structure containing list or dict.\n\n    Args:\n        obj: Module: \n        key: str: \n    "

    def f(_, __, cur):
        if False:
            return 10
        return cur
    return _access_structure(obj, key, callback=f)

def set_expand_structure(obj: Module, key: str, value):
    if False:
        return 10
    "Sets Module's attribute compatible with complex key from Module's :meth:`~.named_children`.\n    Supports handling structure containing list or dict.\n    "

    def f(parent, key, cur):
        if False:
            print('Hello World!')
        if isinstance(parent, (Tensor, Module)):
            if isinstance(cur, Sequential):
                parent[int(key)] = value
            else:
                setattr(parent, key, value)
        else:
            parent[key] = value
    _access_structure(obj, key, callback=f)

@contextlib.contextmanager
def set_module_mode_safe(module: Module, training: bool=False):
    if False:
        return 10
    'Adjust module to training/eval mode temporarily.\n\n    Args:\n        module: used module.\n        training: training (bool): training mode. True for train mode, False fro eval mode.\n    '
    backup_stats = {}

    def recursive_backup_stats(module, mode):
        if False:
            return 10
        for m in module.modules():
            backup_stats[m] = m.training
            m.train(mode, recursive=False)

    def recursive_recover_stats(module):
        if False:
            for i in range(10):
                print('nop')
        for m in module.modules():
            m.training = backup_stats.pop(m)
    recursive_backup_stats(module, mode=training)
    yield module
    recursive_recover_stats(module)