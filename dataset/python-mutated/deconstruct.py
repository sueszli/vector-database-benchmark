from importlib import import_module
from django.utils.version import get_docs_version

def deconstructible(*args, path=None):
    if False:
        return 10
    '\n    Class decorator that allows the decorated class to be serialized\n    by the migrations subsystem.\n\n    The `path` kwarg specifies the import path.\n    '

    def decorator(klass):
        if False:
            return 10

        def __new__(cls, *args, **kwargs):
            if False:
                while True:
                    i = 10
            obj = super(klass, cls).__new__(cls)
            obj._constructor_args = (args, kwargs)
            return obj

        def deconstruct(obj):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return a 3-tuple of class import path, positional arguments,\n            and keyword arguments.\n            '
            if path and type(obj) is klass:
                (module_name, _, name) = path.rpartition('.')
            else:
                module_name = obj.__module__
                name = obj.__class__.__name__
            module = import_module(module_name)
            if not hasattr(module, name):
                raise ValueError('Could not find object %s in %s.\nPlease note that you cannot serialize things like inner classes. Please move the object into the main module body to use migrations.\nFor more information, see https://docs.djangoproject.com/en/%s/topics/migrations/#serializing-values' % (name, module_name, get_docs_version()))
            return (path if path and type(obj) is klass else f'{obj.__class__.__module__}.{name}', obj._constructor_args[0], obj._constructor_args[1])
        klass.__new__ = staticmethod(__new__)
        klass.deconstruct = deconstruct
        return klass
    if not args:
        return decorator
    return decorator(*args)