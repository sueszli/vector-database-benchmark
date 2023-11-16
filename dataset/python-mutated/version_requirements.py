import sys
from packaging import version as _version

def ensure_python_version(min_version):
    if False:
        while True:
            i = 10
    if not isinstance(min_version, tuple):
        min_version = (min_version,)
    if sys.version_info < min_version:
        from platform import python_version
        raise ImportError("\n\nYou are running scikit-image on an unsupported version of Python.\n\nUnfortunately, scikit-image 0.15 and above no longer work with your installed\nversion of Python ({}).  You therefore have two options: either upgrade to\nPython {}, or install an older version of scikit-image.\n\nFor Python 2.7 or Python 3.4, use\n\n $ pip install 'scikit-image<0.15'\n\nPlease also consider updating `pip` and `setuptools`:\n\n $ pip install pip setuptools --upgrade\n\nNewer versions of these tools avoid installing packages incompatible\nwith your version of Python.\n".format(python_version(), '.'.join([str(v) for v in min_version])))

def _check_version(actver, version, cmp_op):
    if False:
        return 10
    '\n    Check version string of an active module against a required version.\n\n    If dev/prerelease tags result in TypeError for string-number comparison,\n    it is assumed that the dependency is satisfied.\n    Users on dev branches are responsible for keeping their own packages up to\n    date.\n    '
    try:
        if cmp_op == '>':
            return _version.parse(actver) > _version.parse(version)
        elif cmp_op == '>=':
            return _version.parse(actver) >= _version.parse(version)
        elif cmp_op == '=':
            return _version.parse(actver) == _version.parse(version)
        elif cmp_op == '<':
            return _version.parse(actver) < _version.parse(version)
        else:
            return False
    except TypeError:
        return True

def get_module_version(module_name):
    if False:
        while True:
            i = 10
    "Return module version or None if version can't be retrieved."
    mod = __import__(module_name, fromlist=[module_name.rpartition('.')[-1]])
    return getattr(mod, '__version__', getattr(mod, 'VERSION', None))

def is_installed(name, version=None):
    if False:
        while True:
            i = 10
    'Test if *name* is installed.\n\n    Parameters\n    ----------\n    name : str\n        Name of module or "python"\n    version : str, optional\n        Version string to test against.\n        If version is not None, checking version\n        (must have an attribute named \'__version__\' or \'VERSION\')\n        Version may start with =, >=, > or < to specify the exact requirement\n\n    Returns\n    -------\n    out : bool\n        True if `name` is installed matching the optional version.\n    '
    if name.lower() == 'python':
        actver = sys.version[:6]
    else:
        try:
            actver = get_module_version(name)
        except ImportError:
            return False
    if version is None:
        return True
    else:
        import re
        match = re.search('[0-9]', version)
        assert match is not None, 'Invalid version number'
        symb = version[:match.start()]
        if not symb:
            symb = '='
        assert symb in ('>=', '>', '=', '<'), f"Invalid version condition '{symb}'"
        version = version[match.start():]
        return _check_version(actver, version, symb)

def require(name, version=None):
    if False:
        return 10
    'Return decorator that forces a requirement for a function or class.\n\n    Parameters\n    ----------\n    name : str\n        Name of module or "python".\n    version : str, optional\n        Version string to test against.\n        If version is not None, checking version\n        (must have an attribute named \'__version__\' or \'VERSION\')\n        Version may start with =, >=, > or < to specify the exact requirement\n\n    Returns\n    -------\n    func : function\n        A decorator that raises an ImportError if a function is run\n        in the absence of the input dependency.\n    '
    import functools

    def decorator(obj):
        if False:
            i = 10
            return i + 15

        @functools.wraps(obj)
        def func_wrapped(*args, **kwargs):
            if False:
                print('Hello World!')
            if is_installed(name, version):
                return obj(*args, **kwargs)
            else:
                msg = f'"{obj}" in "{obj.__module__}" requires "{name}'
                if version is not None:
                    msg += f' {version}'
                raise ImportError(msg + '"')
        return func_wrapped
    return decorator

def get_module(module_name, version=None):
    if False:
        while True:
            i = 10
    "Return a module object of name *module_name* if installed.\n\n    Parameters\n    ----------\n    module_name : str\n        Name of module.\n    version : str, optional\n        Version string to test against.\n        If version is not None, checking version\n        (must have an attribute named '__version__' or 'VERSION')\n        Version may start with =, >=, > or < to specify the exact requirement\n\n    Returns\n    -------\n    mod : module or None\n        Module if *module_name* is installed matching the optional version\n        or None otherwise.\n    "
    if not is_installed(module_name, version):
        return None
    return __import__(module_name, fromlist=[module_name.rpartition('.')[-1]])