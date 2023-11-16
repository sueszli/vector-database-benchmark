"""Utility functions for finding modules

Utility functions for finding modules on sys.path.

"""
import importlib
import sys

def find_mod(module_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find module `module_name` on sys.path, and return the path to module `module_name`.\n\n    * If `module_name` refers to a module directory, then return path to `__init__` file.\n        * If `module_name` is a directory without an __init__file, return None.\n\n    * If module is missing or does not have a `.py` or `.pyw` extension, return None.\n        * Note that we are not interested in running bytecode.\n\n    * Otherwise, return the fill path of the module.\n\n    Parameters\n    ----------\n    module_name : str\n\n    Returns\n    -------\n    module_path : str\n        Path to module `module_name`, its __init__.py, or None,\n        depending on above conditions.\n    '
    spec = importlib.util.find_spec(module_name)
    module_path = spec.origin
    if module_path is None:
        if spec.loader in sys.meta_path:
            return spec.loader
        return None
    else:
        split_path = module_path.split('.')
        if split_path[-1] in ['py', 'pyw']:
            return module_path
        else:
            return None