import importlib
import logging
import re
import sys
from enum import Enum
from tqdm import tqdm
logger = logging.getLogger('featuretools.utils')

def make_tqdm_iterator(**kwargs):
    if False:
        print('Hello World!')
    options = {'file': sys.stdout, 'leave': True}
    options.update(kwargs)
    return tqdm(**options)

def get_relationship_column_id(path):
    if False:
        return 10
    (_, r) = path[0]
    child_link_name = r._child_column_name
    for (_, r) in path[1:]:
        parent_link_name = child_link_name
        child_link_name = '%s.%s' % (r.parent_name, parent_link_name)
    return child_link_name

def find_descendents(cls):
    if False:
        for i in range(10):
            print('nop')
    '\n    A generator which yields all descendent classes of the given class\n    (including the given class)\n\n    Args:\n        cls (Class): the class to find descendents of\n    '
    yield cls
    for sub in cls.__subclasses__():
        for c in find_descendents(sub):
            yield c

def import_or_raise(library, error_msg):
    if False:
        i = 10
        return i + 15
    '\n    Attempts to import the requested library.  If the import fails, raises an\n    ImportErorr with the supplied\n\n    Args:\n        library (str): the name of the library\n        error_msg (str): error message to return if the import fails\n    '
    try:
        return importlib.import_module(library)
    except ImportError:
        raise ImportError(error_msg)

def import_or_none(library):
    if False:
        while True:
            i = 10
    '\n    Attemps to import the requested library.\n\n    Args:\n        library (str): the name of the library\n    Returns: the library if it is installed, else None\n    '
    try:
        return importlib.import_module(library)
    except ImportError:
        return None

def is_instance(obj, modules, classnames):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if the given object is an instance of classname in module(s). Module\n    can be None (i.e. not installed)\n\n    Args:\n        obj (obj): object to test\n        modules (module or tuple[module]): module to check, can be also be None (will be ignored)\n        classnames (str or tuple[str]): classname from module to check. If multiple values are\n                                        provided, they should match with a single module in order.\n                                        If a single value is provided, will be used for all modules.\n    Returns:\n        bool: True if object is an instance of classname from corresponding module, otherwise False.\n              Also returns False if the module is None (i.e. module is not installed)\n    '
    if type(modules) is not tuple:
        modules = (modules,)
    if type(classnames) is not tuple:
        classnames = (classnames,) * len(modules)
    if len(modules) != len(classnames):
        raise ValueError('Number of modules does not match number of classnames')
    to_check = tuple((getattr(mod, classname, mod) for (mod, classname) in zip(modules, classnames) if mod))
    return isinstance(obj, to_check)

def camel_and_title_to_snake(name):
    if False:
        print('Hello World!')
    name = re.sub('([^_\\d]+)(\\d+)', '\\1_\\2', name)
    name = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
    return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', name).lower()

class Library(str, Enum):
    PANDAS = 'pandas'
    DASK = 'Dask'
    SPARK = 'Spark'