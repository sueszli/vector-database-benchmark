"""Helper functions for modules."""
import os
import importlib

def get_parent_dir(module):
    if False:
        for i in range(10):
            print('nop')
    return os.path.abspath(os.path.join(os.path.dirname(module.__file__), '..'))

def get_parent_dir_for_name(module_name):
    if False:
        return 10
    'Get parent directory for module with the given name.\n\n  Args:\n    module_name: Module name for e.g.\n      tensorflow_estimator.python.estimator.api._v1.estimator.\n\n  Returns:\n    Path to the parent directory if module is found and None otherwise.\n    Given example above, it should return:\n      /pathtoestimator/tensorflow_estimator/python/estimator/api/_v1.\n  '
    name_split = module_name.split('.')
    if not name_split:
        return None
    try:
        spec = importlib.util.find_spec(name_split[0])
    except ValueError:
        return None
    if not spec or not spec.origin:
        return None
    base_path = os.path.dirname(spec.origin)
    return os.path.join(base_path, *name_split[1:-1])