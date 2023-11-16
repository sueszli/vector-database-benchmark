import inspect

def is_cython(obj):
    if False:
        while True:
            i = 10
    'Check if an object is a Cython function or method'

    def check_cython(x):
        if False:
            while True:
                i = 10
        return type(x).__name__ == 'cython_function_or_method'
    return check_cython(obj) or (hasattr(obj, '__func__') and check_cython(obj.__func__))

def is_function_or_method(obj):
    if False:
        i = 10
        return i + 15
    'Check if an object is a function or method.\n\n    Args:\n        obj: The Python object in question.\n\n    Returns:\n        True if the object is an function or method.\n    '
    return inspect.isfunction(obj) or inspect.ismethod(obj) or is_cython(obj)

def is_class_method(f):
    if False:
        return 10
    'Returns whether the given method is a class_method.'
    return hasattr(f, '__self__') and f.__self__ is not None

def is_static_method(cls, f_name):
    if False:
        return 10
    'Returns whether the class has a static method with the given name.\n\n    Args:\n        cls: The Python class (i.e. object of type `type`) to\n            search for the method in.\n        f_name: The name of the method to look up in this class\n            and check whether or not it is static.\n    '
    for cls in inspect.getmro(cls):
        if f_name in cls.__dict__:
            return isinstance(cls.__dict__[f_name], staticmethod)
    return False