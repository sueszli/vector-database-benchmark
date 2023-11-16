""" Global constant values.

"""
from nuitka import Options
from nuitka.__past__ import long
from nuitka.plugins.Plugins import Plugins
from nuitka.PythonVersions import python_version
from nuitka.utils.Utils import isWin32Windows

def getConstantDefaultPopulation():
    if False:
        return 10
    'These are values for non-trivial constants.\n\n    Constants that have a direct name, e.g. Py_True are trivial, these are for things that must\n    be constructed through code.\n    '
    result = [(), {}, 0, 1, -1, 0.0, -0.0, 1.0, -1.0, long(0), '', b'', '__module__', '__class__', '__name__', '__package__', '__metaclass__', '__abstractmethods__', '__dict__', '__doc__', '__file__', '__path__', '__enter__', '__exit__', '__builtins__', '__all__', '__init__', '__cmp__', '__iter__', '__loader__', '__compiled__', '__nuitka__', 'inspect', 'compile', 'range', 'open', 'super', 'sum', 'format', '__import__', 'bytearray', 'staticmethod', 'classmethod', 'keys', 'name', 'globals', 'locals', 'fromlist', 'level', 'read', 'rb', '/', '\\', 'path', 'basename', 'abspath', 'isabs', 'exists', 'isdir', 'isfile', 'listdir']
    if python_version < 832:
        result += ('__newobj__',)
    else:
        result += ('getattr',)
    if python_version >= 768:
        result += ('__cached__',)
        result += ('print', 'end', 'file')
        result.append('bytes')
    result.append('.')
    if python_version >= 768:
        result.append('__loader__')
    if python_version >= 832:
        result.append('send')
    if python_version >= 768:
        result += ('throw', 'close')
    if python_version < 768:
        result += ('__getattr__', '__setattr__', '__delattr__')
        result += ('exc_type', 'exc_value', 'exc_traceback')
        result.append('join')
    if python_version < 768:
        result.append('xrange')
    if not Options.shallMakeModule():
        result.append('site')
    if not Options.shallMakeModule():
        result += ('type', 'len', 'range', 'repr', 'int', 'iter')
        if python_version < 768:
            result.append('long')
    if python_version >= 832:
        result += ('__spec__', '_initializing', 'parent')
    if python_version >= 848:
        result.append('types')
    if not Options.shallMakeModule():
        result.append('__main__')
    if python_version >= 912:
        result.append('as_file')
        result.append('register')
    if python_version >= 880:
        result.append('__class_getitem__')
    if python_version >= 880:
        result.append('reconfigure')
        result.append('encoding')
        result.append('line_buffering')
    if python_version >= 928:
        result.append('__match_args__')
        if Options.is_debug:
            result.append('__args__')
    if python_version >= 944:
        result.append('__aenter__')
        result.append('__aexit__')
    if isWin32Windows():
        result.append('fileno')
    for value in Plugins.getExtraConstantDefaultPopulation():
        if value not in result:
            result.append(value)
    return result