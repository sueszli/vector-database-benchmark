"""
Standardized versions of various cool and/or strange things that you can do
with Python's reflection capabilities.
"""
import sys
from jsonschema.compat import PY3

class _NoModuleFound(Exception):
    """
    No module was found because none exists.
    """

class InvalidName(ValueError):
    """
    The given name is not a dot-separated list of Python objects.
    """

class ModuleNotFound(InvalidName):
    """
    The module associated with the given name doesn't exist and it can't be
    imported.
    """

class ObjectNotFound(InvalidName):
    """
    The object associated with the given name doesn't exist and it can't be
    imported.
    """
if PY3:

    def reraise(exception, traceback):
        if False:
            i = 10
            return i + 15
        raise exception.with_traceback(traceback)
else:
    exec('def reraise(exception, traceback):\n        raise exception.__class__, exception, traceback')
reraise.__doc__ = '\nRe-raise an exception, with an optional traceback, in a way that is compatible\nwith both Python 2 and Python 3.\n\nNote that on Python 3, re-raised exceptions will be mutated, with their\nC{__traceback__} attribute being set.\n\n@param exception: The exception instance.\n@param traceback: The traceback to use, or C{None} indicating a new traceback.\n'

def _importAndCheckStack(importName):
    if False:
        i = 10
        return i + 15
    '\n    Import the given name as a module, then walk the stack to determine whether\n    the failure was the module not existing, or some code in the module (for\n    example a dependent import) failing.  This can be helpful to determine\n    whether any actual application code was run.  For example, to distiguish\n    administrative error (entering the wrong module name), from programmer\n    error (writing buggy code in a module that fails to import).\n\n    @param importName: The name of the module to import.\n    @type importName: C{str}\n    @raise Exception: if something bad happens.  This can be any type of\n        exception, since nobody knows what loading some arbitrary code might\n        do.\n    @raise _NoModuleFound: if no module was found.\n    '
    try:
        return __import__(importName)
    except ImportError:
        (excType, excValue, excTraceback) = sys.exc_info()
        while excTraceback:
            execName = excTraceback.tb_frame.f_globals['__name__']
            if execName is None or execName == importName:
                reraise(excValue, excTraceback)
            excTraceback = excTraceback.tb_next
        raise _NoModuleFound()

def namedAny(name):
    if False:
        i = 10
        return i + 15
    "\n    Retrieve a Python object by its fully qualified name from the global Python\n    module namespace.  The first part of the name, that describes a module,\n    will be discovered and imported.  Each subsequent part of the name is\n    treated as the name of an attribute of the object specified by all of the\n    name which came before it.  For example, the fully-qualified name of this\n    object is 'twisted.python.reflect.namedAny'.\n\n    @type name: L{str}\n    @param name: The name of the object to return.\n\n    @raise InvalidName: If the name is an empty string, starts or ends with\n        a '.', or is otherwise syntactically incorrect.\n\n    @raise ModuleNotFound: If the name is syntactically correct but the\n        module it specifies cannot be imported because it does not appear to\n        exist.\n\n    @raise ObjectNotFound: If the name is syntactically correct, includes at\n        least one '.', but the module it specifies cannot be imported because\n        it does not appear to exist.\n\n    @raise AttributeError: If an attribute of an object along the way cannot be\n        accessed, or a module along the way is not found.\n\n    @return: the Python object identified by 'name'.\n    "
    if not name:
        raise InvalidName('Empty module name')
    names = name.split('.')
    if '' in names:
        raise InvalidName("name must be a string giving a '.'-separated list of Python identifiers, not %r" % (name,))
    topLevelPackage = None
    moduleNames = names[:]
    while not topLevelPackage:
        if moduleNames:
            trialname = '.'.join(moduleNames)
            try:
                topLevelPackage = _importAndCheckStack(trialname)
            except _NoModuleFound:
                moduleNames.pop()
        elif len(names) == 1:
            raise ModuleNotFound('No module named %r' % (name,))
        else:
            raise ObjectNotFound('%r does not name an object' % (name,))
    obj = topLevelPackage
    for n in names[1:]:
        obj = getattr(obj, n)
    return obj