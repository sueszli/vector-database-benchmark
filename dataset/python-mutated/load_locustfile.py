import importlib
import inspect
import os
import sys
from typing import Dict, List, Optional, Tuple
from ..shape import LoadTestShape
from ..user import User

def is_user_class(item):
    if False:
        return 10
    '\n    Check if a variable is a runnable (non-abstract) User class\n    '
    return bool(inspect.isclass(item) and issubclass(item, User) and (item.abstract is False))

def is_shape_class(item):
    if False:
        return 10
    '\n    Check if a class is a LoadTestShape\n    '
    return bool(inspect.isclass(item) and issubclass(item, LoadTestShape) and (not getattr(item, 'abstract', True)))

def load_locustfile(path) -> Tuple[Optional[str], Dict[str, User], List[LoadTestShape]]:
    if False:
        while True:
            i = 10
    '\n    Import given locustfile path and return (docstring, callables).\n\n    Specifically, the locustfile\'s ``__doc__`` attribute (a string) and a\n    dictionary of ``{\'name\': callable}`` containing all callables which pass\n    the "is a Locust" test.\n    '
    sys.path.insert(0, os.getcwd())
    (directory, locustfile) = os.path.split(path)
    added_to_path = False
    index = None
    if directory not in sys.path:
        sys.path.insert(0, directory)
        added_to_path = True
    else:
        i = sys.path.index(directory)
        if i != 0:
            index = i
            sys.path.insert(0, directory)
            del sys.path[i + 1]
    source = importlib.machinery.SourceFileLoader(os.path.splitext(locustfile)[0], path)
    imported = source.load_module()
    if added_to_path:
        del sys.path[0]
    if index is not None:
        sys.path.insert(index + 1, directory)
        del sys.path[0]
    user_classes = {name: value for (name, value) in vars(imported).items() if is_user_class(value)}
    shape_classes = [value() for (name, value) in vars(imported).items() if is_shape_class(value)]
    return (imported.__doc__, user_classes, shape_classes)