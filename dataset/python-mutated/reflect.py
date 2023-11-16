"""
Standardized versions of various cool and/or strange things that you can do
with Python's reflection capabilities.
"""
import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
RegexType = type(re.compile(''))

def prefixedMethodNames(classObj, prefix):
    if False:
        print('Hello World!')
    '\n    Given a class object C{classObj}, returns a list of method names that match\n    the string C{prefix}.\n\n    @param classObj: A class object from which to collect method names.\n\n    @param prefix: A native string giving a prefix.  Each method with a name\n        which begins with this prefix will be returned.\n    @type prefix: L{str}\n\n    @return: A list of the names of matching methods of C{classObj} (and base\n        classes of C{classObj}).\n    @rtype: L{list} of L{str}\n    '
    dct = {}
    addMethodNamesToDict(classObj, dct, prefix)
    return list(dct.keys())

def addMethodNamesToDict(classObj, dict, prefix, baseClass=None):
    if False:
        print('Hello World!')
    '\n    This goes through C{classObj} (and its bases) and puts method names\n    starting with \'prefix\' in \'dict\' with a value of 1. if baseClass isn\'t\n    None, methods will only be added if classObj is-a baseClass\n\n    If the class in question has the methods \'prefix_methodname\' and\n    \'prefix_methodname2\', the resulting dict should look something like:\n    {"methodname": 1, "methodname2": 1}.\n\n    @param classObj: A class object from which to collect method names.\n\n    @param dict: A L{dict} which will be updated with the results of the\n        accumulation.  Items are added to this dictionary, with method names as\n        keys and C{1} as values.\n    @type dict: L{dict}\n\n    @param prefix: A native string giving a prefix.  Each method of C{classObj}\n        (and base classes of C{classObj}) with a name which begins with this\n        prefix will be returned.\n    @type prefix: L{str}\n\n    @param baseClass: A class object at which to stop searching upwards for new\n        methods.  To collect all method names, do not pass a value for this\n        parameter.\n\n    @return: L{None}\n    '
    for base in classObj.__bases__:
        addMethodNamesToDict(base, dict, prefix, baseClass)
    if baseClass is None or baseClass in classObj.__bases__:
        for (name, method) in classObj.__dict__.items():
            optName = name[len(prefix):]
            if type(method) is types.FunctionType and name[:len(prefix)] == prefix and len(optName):
                dict[optName] = 1

def prefixedMethods(obj, prefix=''):
    if False:
        i = 10
        return i + 15
    '\n    Given an object C{obj}, returns a list of method objects that match the\n    string C{prefix}.\n\n    @param obj: An arbitrary object from which to collect methods.\n\n    @param prefix: A native string giving a prefix.  Each method of C{obj} with\n        a name which begins with this prefix will be returned.\n    @type prefix: L{str}\n\n    @return: A list of the matching method objects.\n    @rtype: L{list}\n    '
    dct = {}
    accumulateMethods(obj, dct, prefix)
    return list(dct.values())

def accumulateMethods(obj, dict, prefix='', curClass=None):
    if False:
        i = 10
        return i + 15
    '\n    Given an object C{obj}, add all methods that begin with C{prefix}.\n\n    @param obj: An arbitrary object to collect methods from.\n\n    @param dict: A L{dict} which will be updated with the results of the\n        accumulation.  Items are added to this dictionary, with method names as\n        keys and corresponding instance method objects as values.\n    @type dict: L{dict}\n\n    @param prefix: A native string giving a prefix.  Each method of C{obj} with\n        a name which begins with this prefix will be returned.\n    @type prefix: L{str}\n\n    @param curClass: The class in the inheritance hierarchy at which to start\n        collecting methods.  Collection proceeds up.  To collect all methods\n        from C{obj}, do not pass a value for this parameter.\n\n    @return: L{None}\n    '
    if not curClass:
        curClass = obj.__class__
    for base in curClass.__bases__:
        if base is not object:
            accumulateMethods(obj, dict, prefix, base)
    for (name, method) in curClass.__dict__.items():
        optName = name[len(prefix):]
        if type(method) is types.FunctionType and name[:len(prefix)] == prefix and len(optName):
            dict[optName] = getattr(obj, name)

def namedModule(name):
    if False:
        return 10
    '\n    Return a module given its name.\n    '
    topLevel = __import__(name)
    packages = name.split('.')[1:]
    m = topLevel
    for p in packages:
        m = getattr(m, p)
    return m

def namedObject(name):
    if False:
        while True:
            i = 10
    '\n    Get a fully named module-global object.\n    '
    classSplit = name.split('.')
    module = namedModule('.'.join(classSplit[:-1]))
    return getattr(module, classSplit[-1])
namedClass = namedObject

def requireModule(name, default=None):
    if False:
        i = 10
        return i + 15
    '\n    Try to import a module given its name, returning C{default} value if\n    C{ImportError} is raised during import.\n\n    @param name: Module name as it would have been passed to C{import}.\n    @type name: C{str}.\n\n    @param default: Value returned in case C{ImportError} is raised while\n        importing the module.\n\n    @return: Module or default value.\n    '
    try:
        return namedModule(name)
    except ImportError:
        return default

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

def _importAndCheckStack(importName):
    if False:
        print('Hello World!')
    '\n    Import the given name as a module, then walk the stack to determine whether\n    the failure was the module not existing, or some code in the module (for\n    example a dependent import) failing.  This can be helpful to determine\n    whether any actual application code was run.  For example, to distiguish\n    administrative error (entering the wrong module name), from programmer\n    error (writing buggy code in a module that fails to import).\n\n    @param importName: The name of the module to import.\n    @type importName: C{str}\n    @raise Exception: if something bad happens.  This can be any type of\n        exception, since nobody knows what loading some arbitrary code might\n        do.\n    @raise _NoModuleFound: if no module was found.\n    '
    try:
        return __import__(importName)
    except ImportError:
        (excType, excValue, excTraceback) = sys.exc_info()
        while excTraceback:
            execName = excTraceback.tb_frame.f_globals['__name__']
            if execName == importName:
                raise excValue.with_traceback(excTraceback)
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
            raise ModuleNotFound(f'No module named {name!r}')
        else:
            raise ObjectNotFound(f'{name!r} does not name an object')
    obj = topLevelPackage
    for n in names[1:]:
        obj = getattr(obj, n)
    return obj

def filenameToModuleName(fn):
    if False:
        print('Hello World!')
    "\n    Convert a name in the filesystem to the name of the Python module it is.\n\n    This is aggressive about getting a module name back from a file; it will\n    always return a string.  Aggressive means 'sometimes wrong'; it won't look\n    at the Python path or try to do any error checking: don't use this method\n    unless you already know that the filename you're talking about is a Python\n    module.\n\n    @param fn: A filesystem path to a module or package; C{bytes} on Python 2,\n        C{bytes} or C{unicode} on Python 3.\n\n    @return: A hopefully importable module name.\n    @rtype: C{str}\n    "
    if isinstance(fn, bytes):
        initPy = b'__init__.py'
    else:
        initPy = '__init__.py'
    fullName = os.path.abspath(fn)
    base = os.path.basename(fn)
    if not base:
        base = os.path.basename(fn[:-1])
    modName = nativeString(os.path.splitext(base)[0])
    while 1:
        fullName = os.path.dirname(fullName)
        if os.path.exists(os.path.join(fullName, initPy)):
            modName = '{}.{}'.format(nativeString(os.path.basename(fullName)), nativeString(modName))
        else:
            break
    return modName

def qual(clazz: Type[object]) -> str:
    if False:
        return 10
    '\n    Return full import path of a class.\n    '
    return clazz.__module__ + '.' + clazz.__name__

def _determineClass(x):
    if False:
        print('Hello World!')
    try:
        return x.__class__
    except BaseException:
        return type(x)

def _determineClassName(x):
    if False:
        for i in range(10):
            print('nop')
    c = _determineClass(x)
    try:
        return c.__name__
    except BaseException:
        try:
            return str(c)
        except BaseException:
            return '<BROKEN CLASS AT 0x%x>' % id(c)

def _safeFormat(formatter: Union[types.FunctionType, Type[str]], o: object) -> str:
    if False:
        while True:
            i = 10
    '\n    Helper function for L{safe_repr} and L{safe_str}.\n\n    Called when C{repr} or C{str} fail. Returns a string containing info about\n    C{o} and the latest exception.\n\n    @param formatter: C{str} or C{repr}.\n    @type formatter: C{type}\n    @param o: Any object.\n\n    @rtype: C{str}\n    @return: A string containing information about C{o} and the raised\n        exception.\n    '
    io = StringIO()
    traceback.print_exc(file=io)
    className = _determineClassName(o)
    tbValue = io.getvalue()
    return '<{} instance at 0x{:x} with {} error:\n {}>'.format(className, id(o), formatter.__name__, tbValue)

def safe_repr(o):
    if False:
        return 10
    "\n    Returns a string representation of an object, or a string containing a\n    traceback, if that object's __repr__ raised an exception.\n\n    @param o: Any object.\n\n    @rtype: C{str}\n    "
    try:
        return repr(o)
    except BaseException:
        return _safeFormat(repr, o)

def safe_str(o: object) -> str:
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a string representation of an object, or a string containing a\n    traceback, if that object's __str__ raised an exception.\n\n    @param o: Any object.\n    "
    if isinstance(o, bytes):
        try:
            return o.decode('utf-8')
        except BaseException:
            pass
    try:
        return str(o)
    except BaseException:
        return _safeFormat(str, o)

class QueueMethod:
    """
    I represent a method that doesn't exist yet.
    """

    def __init__(self, name, calls):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.calls = calls

    def __call__(self, *args):
        if False:
            print('Hello World!')
        self.calls.append((self.name, args))

def fullFuncName(func):
    if False:
        while True:
            i = 10
    qualName = str(pickle.whichmodule(func, func.__name__)) + '.' + func.__name__
    if namedObject(qualName) is not func:
        raise Exception(f"Couldn't find {func} as {qualName}.")
    return qualName

def getClass(obj):
    if False:
        print('Hello World!')
    "\n    Return the class or type of object 'obj'.\n    "
    return type(obj)

def accumulateClassDict(classObj, attr, adict, baseClass=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Accumulate all attributes of a given name in a class hierarchy into a single dictionary.\n\n    Assuming all class attributes of this name are dictionaries.\n    If any of the dictionaries being accumulated have the same key, the\n    one highest in the class hierarchy wins.\n    (XXX: If "highest" means "closest to the starting class".)\n\n    Ex::\n\n      class Soy:\n        properties = {"taste": "bland"}\n\n      class Plant:\n        properties = {"colour": "green"}\n\n      class Seaweed(Plant):\n        pass\n\n      class Lunch(Soy, Seaweed):\n        properties = {"vegan": 1 }\n\n      dct = {}\n\n      accumulateClassDict(Lunch, "properties", dct)\n\n      print(dct)\n\n    {"taste": "bland", "colour": "green", "vegan": 1}\n    '
    for base in classObj.__bases__:
        accumulateClassDict(base, attr, adict)
    if baseClass is None or baseClass in classObj.__bases__:
        adict.update(classObj.__dict__.get(attr, {}))

def accumulateClassList(classObj, attr, listObj, baseClass=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Accumulate all attributes of a given name in a class hierarchy into a single list.\n\n    Assuming all class attributes of this name are lists.\n    '
    for base in classObj.__bases__:
        accumulateClassList(base, attr, listObj)
    if baseClass is None or baseClass in classObj.__bases__:
        listObj.extend(classObj.__dict__.get(attr, []))

def isSame(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a is b

def isLike(a, b):
    if False:
        while True:
            i = 10
    return a == b

def modgrep(goal):
    if False:
        return 10
    return objgrep(sys.modules, goal, isLike, 'sys.modules')

def isOfType(start, goal):
    if False:
        while True:
            i = 10
    return type(start) is goal

def findInstances(start, t):
    if False:
        while True:
            i = 10
    return objgrep(start, t, isOfType)

def objgrep(start, goal, eq=isLike, path='', paths=None, seen=None, showUnknowns=0, maxDepth=None):
    if False:
        print('Hello World!')
    '\n    L{objgrep} finds paths between C{start} and C{goal}.\n\n    Starting at the python object C{start}, we will loop over every reachable\n    reference, tring to find the python object C{goal} (i.e. every object\n    C{candidate} for whom C{eq(candidate, goal)} is truthy), and return a\n    L{list} of L{str}, where each L{str} is Python syntax for a path between\n    C{start} and C{goal}.\n\n    Since this can be slightly difficult to visualize, here\'s an example::\n\n        >>> class Holder:\n        ...     def __init__(self, x):\n        ...         self.x = x\n        ...\n        >>> start = Holder({"irrelevant": "ignore",\n        ...                 "relevant": [7, 1, 3, 5, 7]})\n        >>> for path in objgrep(start, 7):\n        ...     print("start" + path)\n        start.x[\'relevant\'][0]\n        start.x[\'relevant\'][4]\n\n    This can be useful, for example, when debugging stateful graphs of objects\n    attached to a socket, trying to figure out where a particular connection is\n    attached.\n\n    @param start: The object to start looking at.\n\n    @param goal: The object to search for.\n\n    @param eq: A 2-argument predicate which takes an object found by traversing\n        references starting at C{start}, as well as C{goal}, and returns a\n        boolean.\n\n    @param path: The prefix of the path to include in every return value; empty\n        by default.\n\n    @param paths: The result object to append values to; a list of strings.\n\n    @param seen: A dictionary mapping ints (object IDs) to objects already\n        seen.\n\n    @param showUnknowns: if true, print a message to C{stdout} when\n        encountering objects that C{objgrep} does not know how to traverse.\n\n    @param maxDepth: The maximum number of object references to attempt\n        traversing before giving up.  If an integer, limit to that many links,\n        if C{None}, unlimited.\n\n    @return: A list of strings representing python object paths starting at\n        C{start} and terminating at C{goal}.\n    '
    if paths is None:
        paths = []
    if seen is None:
        seen = {}
    if eq(start, goal):
        paths.append(path)
    if id(start) in seen:
        if seen[id(start)] is start:
            return
    if maxDepth is not None:
        if maxDepth == 0:
            return
        maxDepth -= 1
    seen[id(start)] = start
    args = (paths, seen, showUnknowns, maxDepth)
    if isinstance(start, dict):
        for (k, v) in start.items():
            objgrep(k, goal, eq, path + '{' + repr(v) + '}', *args)
            objgrep(v, goal, eq, path + '[' + repr(k) + ']', *args)
    elif isinstance(start, (list, tuple, deque)):
        for (idx, _elem) in enumerate(start):
            objgrep(start[idx], goal, eq, path + '[' + str(idx) + ']', *args)
    elif isinstance(start, types.MethodType):
        objgrep(start.__self__, goal, eq, path + '.__self__', *args)
        objgrep(start.__func__, goal, eq, path + '.__func__', *args)
        objgrep(start.__self__.__class__, goal, eq, path + '.__self__.__class__', *args)
    elif hasattr(start, '__dict__'):
        for (k, v) in start.__dict__.items():
            objgrep(v, goal, eq, path + '.' + k, *args)
    elif isinstance(start, weakref.ReferenceType):
        objgrep(start(), goal, eq, path + '()', *args)
    elif isinstance(start, (str, int, types.FunctionType, types.BuiltinMethodType, RegexType, float, type(None), IOBase)) or type(start).__name__ in ('wrapper_descriptor', 'method_descriptor', 'member_descriptor', 'getset_descriptor'):
        pass
    elif showUnknowns:
        print('unknown type', type(start), start)
    return paths
__all__ = ['InvalidName', 'ModuleNotFound', 'ObjectNotFound', 'QueueMethod', 'namedModule', 'namedObject', 'namedClass', 'namedAny', 'requireModule', 'safe_repr', 'safe_str', 'prefixedMethodNames', 'addMethodNamesToDict', 'prefixedMethods', 'accumulateMethods', 'fullFuncName', 'qual', 'getClass', 'accumulateClassDict', 'accumulateClassList', 'isSame', 'isLike', 'modgrep', 'isOfType', 'findInstances', 'objgrep', 'filenameToModuleName', 'fullyQualifiedName']
__all__.remove('objgrep')