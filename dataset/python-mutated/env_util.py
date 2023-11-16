import os
import platform
import re
import sys
_system = platform.system()
IS_WINDOWS = _system == 'Windows'
IS_DARWIN = _system == 'Darwin'
IS_LINUX_OR_BSD = _system == 'Linux' or 'BSD' in _system

def is_pex():
    if False:
        print('Hello World!')
    "Return if streamlit running in pex.\n\n    Pex modifies sys.path so the pex file is the first path and that's\n    how we determine we're running in the pex file.\n    "
    if re.match('.*pex$', sys.path[0]):
        return True
    return False

def is_repl():
    if False:
        while True:
            i = 10
    'Return True if running in the Python REPL.'
    import inspect
    root_frame = inspect.stack()[-1]
    filename = root_frame[1]
    if filename.endswith(os.path.join('bin', 'ipython')):
        return True
    if filename in ('<stdin>', '<string>'):
        return True
    return False

def is_executable_in_path(name):
    if False:
        while True:
            i = 10
    'Check if executable is in OS path.'
    from distutils.spawn import find_executable
    return find_executable(name) is not None