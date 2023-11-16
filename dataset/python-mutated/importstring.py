"""
A simple utility to import something by its string name.
"""

def import_item(name):
    if False:
        return 10
    'Import and return ``bar`` given the string ``foo.bar``.\n\n    Calling ``bar = import_item("foo.bar")`` is the functional equivalent of\n    executing the code ``from foo import bar``.\n\n    Parameters\n    ----------\n    name : string\n        The fully qualified name of the module/package being imported.\n\n    Returns\n    -------\n    mod : module object\n        The module that was imported.\n    '
    parts = name.rsplit('.', 1)
    if len(parts) == 2:
        (package, obj) = parts
        module = __import__(package, fromlist=[obj])
        try:
            pak = getattr(module, obj)
        except AttributeError as e:
            raise ImportError('No module named %s' % obj) from e
        return pak
    else:
        return __import__(parts[0])