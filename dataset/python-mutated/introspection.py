"""Functions related to Python runtime introspection."""
import collections
import importlib
import inspect
import os
import sys
import types
from importlib import metadata
from packaging.version import Version
__all__ = ['resolve_name', 'minversion', 'find_current_module', 'isinstancemethod']
__doctest_skip__ = ['find_current_module']
if sys.version_info[:2] >= (3, 10):
    from importlib.metadata import packages_distributions
else:

    def packages_distributions():
        if False:
            print('Hello World!')
        '\n        Return a mapping of top-level packages to their distributions.\n        Note: copied from https://github.com/python/importlib_metadata/pull/287.\n        '
        pkg_to_dist = collections.defaultdict(list)
        for dist in metadata.distributions():
            for pkg in (dist.read_text('top_level.txt') or '').split():
                pkg_to_dist[pkg].append(dist.metadata['Name'])
        return dict(pkg_to_dist)

def resolve_name(name, *additional_parts):
    if False:
        print('Hello World!')
    "Resolve a name like ``module.object`` to an object and return it.\n\n    This ends up working like ``from module import object`` but is easier\n    to deal with than the `__import__` builtin and supports digging into\n    submodules.\n\n    Parameters\n    ----------\n    name : `str`\n        A dotted path to a Python object--that is, the name of a function,\n        class, or other object in a module with the full path to that module,\n        including parent modules, separated by dots.  Also known as the fully\n        qualified name of the object.\n\n    additional_parts : iterable, optional\n        If more than one positional arguments are given, those arguments are\n        automatically dotted together with ``name``.\n\n    Examples\n    --------\n    >>> resolve_name('astropy.utils.introspection.resolve_name')\n    <function resolve_name at 0x...>\n    >>> resolve_name('astropy', 'utils', 'introspection', 'resolve_name')\n    <function resolve_name at 0x...>\n\n    Raises\n    ------\n    `ImportError`\n        If the module or named object is not found.\n    "
    additional_parts = '.'.join(additional_parts)
    if additional_parts:
        name = name + '.' + additional_parts
    parts = name.split('.')
    if len(parts) == 1:
        cursor = 1
        fromlist = []
    else:
        cursor = len(parts) - 1
        fromlist = [parts[-1]]
    module_name = parts[:cursor]
    while cursor > 0:
        try:
            ret = __import__('.'.join(module_name), fromlist=fromlist)
            break
        except ImportError:
            if cursor == 0:
                raise
            cursor -= 1
            module_name = parts[:cursor]
            fromlist = [parts[cursor]]
            ret = ''
    for part in parts[cursor:]:
        try:
            ret = getattr(ret, part)
        except AttributeError:
            raise ImportError(name)
    return ret

def minversion(module, version, inclusive=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns `True` if the specified Python module satisfies a minimum version\n    requirement, and `False` if not.\n\n    Parameters\n    ----------\n    module : module or `str`\n        An imported module of which to check the version, or the name of\n        that module (in which case an import of that module is attempted--\n        if this fails `False` is returned).\n\n    version : `str`\n        The version as a string that this module must have at a minimum (e.g.\n        ``'0.12'``).\n\n    inclusive : `bool`\n        The specified version meets the requirement inclusively (i.e. ``>=``)\n        as opposed to strictly greater than (default: `True`).\n\n    Examples\n    --------\n    >>> import astropy\n    >>> minversion(astropy, '0.4.4')\n    True\n    "
    if isinstance(module, types.ModuleType):
        module_name = module.__name__
        module_version = getattr(module, '__version__', None)
    elif isinstance(module, str):
        module_name = module
        module_version = None
        try:
            module = resolve_name(module_name)
        except ImportError:
            return False
    else:
        raise ValueError(f'module argument must be an actual imported module, or the import name of the module; got {repr(module)}')
    if module_version is None:
        try:
            module_version = metadata.version(module_name)
        except metadata.PackageNotFoundError:
            dist_names = packages_distributions()
            module_version = metadata.version(dist_names[module_name][0])
    if inclusive:
        return Version(module_version) >= Version(version)
    else:
        return Version(module_version) > Version(version)

def find_current_module(depth=1, finddiff=False):
    if False:
        return 10
    "\n    Determines the module/package from which this function is called.\n\n    This function has two modes, determined by the ``finddiff`` option. it\n    will either simply go the requested number of frames up the call\n    stack (if ``finddiff`` is False), or it will go up the call stack until\n    it reaches a module that is *not* in a specified set.\n\n    Parameters\n    ----------\n    depth : int\n        Specifies how far back to go in the call stack (0-indexed, so that\n        passing in 0 gives back `astropy.utils.misc`).\n    finddiff : bool or list\n        If False, the returned ``mod`` will just be ``depth`` frames up from\n        the current frame. Otherwise, the function will start at a frame\n        ``depth`` up from current, and continue up the call stack to the\n        first module that is *different* from those in the provided list.\n        In this case, ``finddiff`` can be a list of modules or modules\n        names. Alternatively, it can be True, which will use the module\n        ``depth`` call stack frames up as the module the returned module\n        most be different from.\n\n    Returns\n    -------\n    mod : module or None\n        The module object or None if the package cannot be found. The name of\n        the module is available as the ``__name__`` attribute of the returned\n        object (if it isn't None).\n\n    Raises\n    ------\n    ValueError\n        If ``finddiff`` is a list with an invalid entry.\n\n    Examples\n    --------\n    The examples below assume that there are two modules in a package named\n    ``pkg``. ``mod1.py``::\n\n        def find1():\n            from astropy.utils import find_current_module\n            print find_current_module(1).__name__\n        def find2():\n            from astropy.utils import find_current_module\n            cmod = find_current_module(2)\n            if cmod is None:\n                print 'None'\n            else:\n                print cmod.__name__\n        def find_diff():\n            from astropy.utils import find_current_module\n            print find_current_module(0,True).__name__\n\n    ``mod2.py``::\n\n        def find():\n            from .mod1 import find2\n            find2()\n\n    With these modules in place, the following occurs::\n\n        >>> from pkg import mod1, mod2\n        >>> from astropy.utils import find_current_module\n        >>> mod1.find1()\n        pkg.mod1\n        >>> mod1.find2()\n        None\n        >>> mod2.find()\n        pkg.mod2\n        >>> find_current_module(0)\n        <module 'astropy.utils.misc' from 'astropy/utils/misc.py'>\n        >>> mod1.find_diff()\n        pkg.mod1\n\n    "
    frm = inspect.currentframe()
    for i in range(depth):
        frm = frm.f_back
        if frm is None:
            return None
    if finddiff:
        currmod = _get_module_from_frame(frm)
        if finddiff is True:
            diffmods = [currmod]
        else:
            diffmods = []
            for fd in finddiff:
                if inspect.ismodule(fd):
                    diffmods.append(fd)
                elif isinstance(fd, str):
                    diffmods.append(importlib.import_module(fd))
                elif fd is True:
                    diffmods.append(currmod)
                else:
                    raise ValueError('invalid entry in finddiff')
        while frm:
            frmb = frm.f_back
            modb = _get_module_from_frame(frmb)
            if modb not in diffmods:
                return modb
            frm = frmb
    else:
        return _get_module_from_frame(frm)

def _get_module_from_frame(frm):
    if False:
        return 10
    "Uses inspect.getmodule() to get the module that the current frame's\n    code is running in.\n\n    However, this does not work reliably for code imported from a zip file,\n    so this provides a fallback mechanism for that case which is less\n    reliable in general, but more reliable than inspect.getmodule() for this\n    particular case.\n    "
    mod = inspect.getmodule(frm)
    if mod is not None:
        return mod
    if '__file__' in frm.f_globals and '__name__' in frm.f_globals:
        filename = frm.f_globals['__file__']
        if filename[-4:].lower() in ('.pyc', '.pyo'):
            filename = filename[:-4] + '.py'
        filename = os.path.realpath(os.path.abspath(filename))
        if filename in inspect.modulesbyfile:
            return sys.modules.get(inspect.modulesbyfile[filename])
        if filename.lower() in inspect.modulesbyfile:
            return sys.modules.get(inspect.modulesbyfile[filename.lower()])
    return None

def find_mod_objs(modname, onlylocals=False):
    if False:
        while True:
            i = 10
    'Returns all the public attributes of a module referenced by name.\n\n    .. note::\n        The returned list *not* include subpackages or modules of\n        ``modname``, nor does it include private attributes (those that\n        begin with \'_\' or are not in `__all__`).\n\n    Parameters\n    ----------\n    modname : str\n        The name of the module to search.\n    onlylocals : bool or list of str\n        If `True`, only attributes that are either members of ``modname`` OR\n        one of its modules or subpackages will be included. If it is a list\n        of strings, those specify the possible packages that will be\n        considered "local".\n\n    Returns\n    -------\n    localnames : list of str\n        A list of the names of the attributes as they are named in the\n        module ``modname`` .\n    fqnames : list of str\n        A list of the full qualified names of the attributes (e.g.,\n        ``astropy.utils.introspection.find_mod_objs``). For attributes that are\n        simple variables, this is based on the local name, but for functions or\n        classes it can be different if they are actually defined elsewhere and\n        just referenced in ``modname``.\n    objs : list of objects\n        A list of the actual attributes themselves (in the same order as\n        the other arguments)\n\n    '
    mod = resolve_name(modname)
    if hasattr(mod, '__all__'):
        pkgitems = [(k, mod.__dict__[k]) for k in mod.__all__]
    else:
        pkgitems = [(k, mod.__dict__[k]) for k in dir(mod) if k[0] != '_']
    ismodule = inspect.ismodule
    localnames = [k for (k, v) in pkgitems if not ismodule(v)]
    objs = [v for (k, v) in pkgitems if not ismodule(v)]
    fqnames = []
    for (obj, lnm) in zip(objs, localnames):
        if hasattr(obj, '__module__') and hasattr(obj, '__name__'):
            fqnames.append(obj.__module__ + '.' + obj.__name__)
        else:
            fqnames.append(modname + '.' + lnm)
    if onlylocals:
        if onlylocals is True:
            onlylocals = [modname]
        valids = [any((fqn.startswith(nm) for nm in onlylocals)) for fqn in fqnames]
        localnames = [e for (i, e) in enumerate(localnames) if valids[i]]
        fqnames = [e for (i, e) in enumerate(fqnames) if valids[i]]
        objs = [e for (i, e) in enumerate(objs) if valids[i]]
    return (localnames, fqnames, objs)

def isinstancemethod(cls, obj):
    if False:
        return 10
    '\n    Returns `True` if the given object is an instance method of the class\n    it is defined on (as opposed to a `staticmethod` or a `classmethod`).\n\n    This requires both the class the object is a member of as well as the\n    object itself in order to make this determination.\n\n    Parameters\n    ----------\n    cls : `type`\n        The class on which this method was defined.\n    obj : `object`\n        A member of the provided class (the membership is not checked directly,\n        but this function will always return `False` if the given object is not\n        a member of the given class).\n\n    Examples\n    --------\n    >>> class MetaClass(type):\n    ...     def a_classmethod(cls): pass\n    ...\n    >>> class MyClass(metaclass=MetaClass):\n    ...     def an_instancemethod(self): pass\n    ...\n    ...     @classmethod\n    ...     def another_classmethod(cls): pass\n    ...\n    ...     @staticmethod\n    ...     def a_staticmethod(): pass\n    ...\n    >>> isinstancemethod(MyClass, MyClass.a_classmethod)\n    False\n    >>> isinstancemethod(MyClass, MyClass.another_classmethod)\n    False\n    >>> isinstancemethod(MyClass, MyClass.a_staticmethod)\n    False\n    >>> isinstancemethod(MyClass, MyClass.an_instancemethod)\n    True\n    '
    return _isinstancemethod(cls, obj)

def _isinstancemethod(cls, obj):
    if False:
        print('Hello World!')
    if not isinstance(obj, types.FunctionType):
        return False
    name = obj.__name__
    for basecls in cls.mro():
        if name in basecls.__dict__:
            return not isinstance(basecls.__dict__[name], staticmethod)
    raise AttributeError(name)