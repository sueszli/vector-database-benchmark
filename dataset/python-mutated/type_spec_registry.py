"""Registry for custom TypeSpecs."""
import re
from tensorflow.python.types import internal
_TYPE_SPEC_TO_NAME = {}
_NAME_TO_TYPE_SPEC = {}
_REGISTERED_NAME_RE = re.compile('^(\\w+\\.)+\\w+$')

def register(name):
    if False:
        for i in range(10):
            print('nop')
    'Decorator used to register a globally unique name for a TypeSpec subclass.\n\n  Args:\n    name: The name of the type spec.  Must be globally unique.  Must have the\n      form `"{project_name}.{type_name}"`.  E.g. `"my_project.MyTypeSpec"`.\n\n  Returns:\n    A class decorator that registers the decorated class with the given name.\n  '
    if not isinstance(name, str):
        raise TypeError('Expected `name` to be a string; got %r' % (name,))
    if not _REGISTERED_NAME_RE.match(name):
        raise ValueError("Registered name must have the form '{project_name}.{type_name}' (e.g. 'my_project.MyTypeSpec'); got %r." % name)

    def decorator_fn(cls):
        if False:
            return 10
        if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
            raise TypeError('Expected `cls` to be a TypeSpec; got %r' % (cls,))
        if cls in _TYPE_SPEC_TO_NAME:
            raise ValueError('Class %s.%s has already been registered with name %s.' % (cls.__module__, cls.__name__, _TYPE_SPEC_TO_NAME[cls]))
        if name in _NAME_TO_TYPE_SPEC:
            raise ValueError('Name %s has already been registered for class %s.%s.' % (name, _NAME_TO_TYPE_SPEC[name].__module__, _NAME_TO_TYPE_SPEC[name].__name__))
        _TYPE_SPEC_TO_NAME[cls] = name
        _NAME_TO_TYPE_SPEC[name] = cls
        return cls
    return decorator_fn

def get_name(cls):
    if False:
        i = 10
        return i + 15
    'Returns the registered name for TypeSpec `cls`.'
    if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
        raise TypeError('Expected `cls` to be a TypeSpec; got %r' % (cls,))
    if cls not in _TYPE_SPEC_TO_NAME:
        raise ValueError('TypeSpec %s.%s has not been registered.' % (cls.__module__, cls.__name__))
    return _TYPE_SPEC_TO_NAME[cls]

def lookup(name):
    if False:
        i = 10
        return i + 15
    'Returns the TypeSpec that has been registered with name `name`.'
    if not isinstance(name, str):
        raise TypeError('Expected `name` to be a string; got %r' % (name,))
    if name not in _NAME_TO_TYPE_SPEC:
        raise ValueError('No TypeSpec has been registered with name %r' % (name,))
    return _NAME_TO_TYPE_SPEC[name]