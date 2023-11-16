"""Tools to assist importing optional external modules."""
import sys
import re
WARN_NOT_INSTALLED = None
WARN_OLD_VERSION = None

def __sympy_debug():
    if False:
        while True:
            i = 10
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError('unrecognized value for SYMPY_DEBUG: %s' % debug_str)
if __sympy_debug():
    WARN_OLD_VERSION = True
    WARN_NOT_INSTALLED = True
_component_re = re.compile('(\\d+ | [a-z]+ | \\.)', re.VERBOSE)

def version_tuple(vstring):
    if False:
        while True:
            i = 10
    components = []
    for x in _component_re.split(vstring):
        if x and x != '.':
            try:
                x = int(x)
            except ValueError:
                pass
            components.append(x)
    return tuple(components)

def import_module(module, min_module_version=None, min_python_version=None, warn_not_installed=None, warn_old_version=None, module_version_attr='__version__', module_version_attr_call_args=None, import_kwargs={}, catch=()):
    if False:
        print('Hello World!')
    "\n    Import and return a module if it is installed.\n\n    If the module is not installed, it returns None.\n\n    A minimum version for the module can be given as the keyword argument\n    min_module_version.  This should be comparable against the module version.\n    By default, module.__version__ is used to get the module version.  To\n    override this, set the module_version_attr keyword argument.  If the\n    attribute of the module to get the version should be called (e.g.,\n    module.version()), then set module_version_attr_call_args to the args such\n    that module.module_version_attr(*module_version_attr_call_args) returns the\n    module's version.\n\n    If the module version is less than min_module_version using the Python <\n    comparison, None will be returned, even if the module is installed. You can\n    use this to keep from importing an incompatible older version of a module.\n\n    You can also specify a minimum Python version by using the\n    min_python_version keyword argument.  This should be comparable against\n    sys.version_info.\n\n    If the keyword argument warn_not_installed is set to True, the function will\n    emit a UserWarning when the module is not installed.\n\n    If the keyword argument warn_old_version is set to True, the function will\n    emit a UserWarning when the library is installed, but cannot be imported\n    because of the min_module_version or min_python_version options.\n\n    Note that because of the way warnings are handled, a warning will be\n    emitted for each module only once.  You can change the default warning\n    behavior by overriding the values of WARN_NOT_INSTALLED and WARN_OLD_VERSION\n    in sympy.external.importtools.  By default, WARN_NOT_INSTALLED is False and\n    WARN_OLD_VERSION is True.\n\n    This function uses __import__() to import the module.  To pass additional\n    options to __import__(), use the import_kwargs keyword argument.  For\n    example, to import a submodule A.B, you must pass a nonempty fromlist option\n    to __import__.  See the docstring of __import__().\n\n    This catches ImportError to determine if the module is not installed.  To\n    catch additional errors, pass them as a tuple to the catch keyword\n    argument.\n\n    Examples\n    ========\n\n    >>> from sympy.external import import_module\n\n    >>> numpy = import_module('numpy')\n\n    >>> numpy = import_module('numpy', min_python_version=(2, 7),\n    ... warn_old_version=False)\n\n    >>> numpy = import_module('numpy', min_module_version='1.5',\n    ... warn_old_version=False) # numpy.__version__ is a string\n\n    >>> # gmpy does not have __version__, but it does have gmpy.version()\n\n    >>> gmpy = import_module('gmpy', min_module_version='1.14',\n    ... module_version_attr='version', module_version_attr_call_args=(),\n    ... warn_old_version=False)\n\n    >>> # To import a submodule, you must pass a nonempty fromlist to\n    >>> # __import__().  The values do not matter.\n    >>> p3 = import_module('mpl_toolkits.mplot3d',\n    ... import_kwargs={'fromlist':['something']})\n\n    >>> # matplotlib.pyplot can raise RuntimeError when the display cannot be opened\n    >>> matplotlib = import_module('matplotlib',\n    ... import_kwargs={'fromlist':['pyplot']}, catch=(RuntimeError,))\n\n    "
    warn_old_version = WARN_OLD_VERSION if WARN_OLD_VERSION is not None else warn_old_version or True
    warn_not_installed = WARN_NOT_INSTALLED if WARN_NOT_INSTALLED is not None else warn_not_installed or False
    import warnings
    if min_python_version:
        if sys.version_info < min_python_version:
            if warn_old_version:
                warnings.warn('Python version is too old to use %s (%s or newer required)' % (module, '.'.join(map(str, min_python_version))), UserWarning, stacklevel=2)
            return
    try:
        mod = __import__(module, **import_kwargs)
        from_list = import_kwargs.get('fromlist', ())
        for submod in from_list:
            if submod == 'collections' and mod.__name__ == 'matplotlib':
                __import__(module + '.' + submod)
    except ImportError:
        if warn_not_installed:
            warnings.warn('%s module is not installed' % module, UserWarning, stacklevel=2)
        return
    except catch as e:
        if warn_not_installed:
            warnings.warn('%s module could not be used (%s)' % (module, repr(e)), stacklevel=2)
        return
    if min_module_version:
        modversion = getattr(mod, module_version_attr)
        if module_version_attr_call_args is not None:
            modversion = modversion(*module_version_attr_call_args)
        if version_tuple(modversion) < version_tuple(min_module_version):
            if warn_old_version:
                if isinstance(min_module_version, str):
                    verstr = min_module_version
                elif isinstance(min_module_version, (tuple, list)):
                    verstr = '.'.join(map(str, min_module_version))
                else:
                    verstr = str(min_module_version)
                warnings.warn('%s version is too old to use (%s or newer required)' % (module, verstr), UserWarning, stacklevel=2)
            return
    return mod