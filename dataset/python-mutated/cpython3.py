from __future__ import annotations
import abc
import fnmatch
from itertools import chain
from operator import methodcaller as method
from pathlib import Path
from textwrap import dedent
from virtualenv.create.describe import Python3Supports
from virtualenv.create.via_global_ref.builtin.ref import PathRefToDest
from virtualenv.create.via_global_ref.store import is_store_python
from .common import CPython, CPythonPosix, CPythonWindows, is_mac_os_framework, is_macos_brew

class CPython3(CPython, Python3Supports, metaclass=abc.ABCMeta):
    """CPython 3 or later."""

class CPython3Posix(CPythonPosix, CPython3):

    @classmethod
    def can_describe(cls, interpreter):
        if False:
            print('Hello World!')
        return is_mac_os_framework(interpreter) is False and is_macos_brew(interpreter) is False and super().can_describe(interpreter)

    def env_patch_text(self):
        if False:
            for i in range(10):
                print('nop')
        text = super().env_patch_text()
        if self.pyvenv_launch_patch_active(self.interpreter):
            text += dedent('\n                # for https://github.com/python/cpython/pull/9516, see https://github.com/pypa/virtualenv/issues/1704\n                import os\n                if "__PYVENV_LAUNCHER__" in os.environ:\n                    del os.environ["__PYVENV_LAUNCHER__"]\n                ')
        return text

    @classmethod
    def pyvenv_launch_patch_active(cls, interpreter):
        if False:
            i = 10
            return i + 15
        ver = interpreter.version_info
        return interpreter.platform == 'darwin' and ((3, 7, 8) > ver >= (3, 7) or (3, 8, 3) > ver >= (3, 8))

class CPython3Windows(CPythonWindows, CPython3):
    """CPython 3 on Windows."""

    @classmethod
    def setup_meta(cls, interpreter):
        if False:
            while True:
                i = 10
        if is_store_python(interpreter):
            return None
        return super().setup_meta(interpreter)

    @classmethod
    def sources(cls, interpreter):
        if False:
            i = 10
            return i + 15
        if cls.has_shim(interpreter):
            refs = cls.executables(interpreter)
        else:
            refs = chain(cls.executables(interpreter), cls.dll_and_pyd(interpreter), cls.python_zip(interpreter))
        yield from refs

    @classmethod
    def executables(cls, interpreter):
        if False:
            for i in range(10):
                print('nop')
        return super().sources(interpreter)

    @classmethod
    def has_shim(cls, interpreter):
        if False:
            return 10
        return interpreter.version_info.minor >= 7 and cls.shim(interpreter) is not None

    @classmethod
    def shim(cls, interpreter):
        if False:
            print('Hello World!')
        shim = Path(interpreter.system_stdlib) / 'venv' / 'scripts' / 'nt' / 'python.exe'
        if shim.exists():
            return shim
        return None

    @classmethod
    def host_python(cls, interpreter):
        if False:
            while True:
                i = 10
        if cls.has_shim(interpreter):
            return cls.shim(interpreter)
        return super().host_python(interpreter)

    @classmethod
    def dll_and_pyd(cls, interpreter):
        if False:
            i = 10
            return i + 15
        folders = [Path(interpreter.system_executable).parent]
        dll_folder = Path(interpreter.system_prefix) / 'DLLs'
        if dll_folder.is_dir():
            folders.append(dll_folder)
        for folder in folders:
            for file in folder.iterdir():
                if file.suffix in ('.pyd', '.dll'):
                    yield PathRefToDest(file, cls.to_bin)

    @classmethod
    def python_zip(cls, interpreter):
        if False:
            i = 10
            return i + 15
        '\n        "python{VERSION}.zip" contains compiled *.pyc std lib packages, where\n        "VERSION" is `py_version_nodot` var from the `sysconfig` module.\n        :see: https://docs.python.org/3/using/windows.html#the-embeddable-package\n        :see: `discovery.py_info.PythonInfo` class (interpreter).\n        :see: `python -m sysconfig` output.\n\n        :note: The embeddable Python distribution for Windows includes\n        "python{VERSION}.zip" and "python{VERSION}._pth" files. User can\n        move/rename *zip* file and edit `sys.path` by editing *_pth* file.\n        Here the `pattern` is used only for the default *zip* file name!\n        '
        pattern = f'*python{interpreter.version_nodot}.zip'
        matches = fnmatch.filter(interpreter.path, pattern)
        matched_paths = map(Path, matches)
        existing_paths = filter(method('exists'), matched_paths)
        path = next(existing_paths, None)
        if path is not None:
            yield PathRefToDest(path, cls.to_bin)
__all__ = ['CPython3', 'CPython3Posix', 'CPython3Windows']