"""
Set search path for pywin32 DLLs. Due to the large number of pywin32 modules, we use a single loader-level script
instead of per-module runtime hook scripts.
"""
import os
import sys

def install():
    if False:
        for i in range(10):
            print('nop')
    pywin32_ext_paths = ('win32', 'pythonwin')
    pywin32_ext_paths = [os.path.join(sys._MEIPASS, pywin32_ext_path) for pywin32_ext_path in pywin32_ext_paths]
    pywin32_ext_paths = [path for path in pywin32_ext_paths if os.path.isdir(path)]
    sys.path.extend(pywin32_ext_paths)
    pywin32_system32_path = os.path.join(sys._MEIPASS, 'pywin32_system32')
    if not os.path.isdir(pywin32_system32_path):
        return
    sys.path.append(pywin32_system32_path)
    os.add_dll_directory(pywin32_system32_path)
    path = os.environ.get('PATH', None)
    if not path:
        path = pywin32_system32_path
    else:
        path = pywin32_system32_path + os.pathsep + path
    os.environ['PATH'] = path