from __future__ import annotations
import abc
from pathlib import Path
from virtualenv.create.describe import PosixSupports, Python3Supports, WindowsSupports
from virtualenv.create.via_global_ref.builtin.ref import PathRefToDest
from .common import PyPy

class PyPy3(PyPy, Python3Supports, metaclass=abc.ABCMeta):

    @classmethod
    def exe_stem(cls):
        if False:
            while True:
                i = 10
        return 'pypy3'

    @classmethod
    def exe_names(cls, interpreter):
        if False:
            while True:
                i = 10
        return super().exe_names(interpreter) | {'pypy'}

class PyPy3Posix(PyPy3, PosixSupports):
    """PyPy 3 on POSIX."""

    @classmethod
    def _shared_libs(cls, python_dir):
        if False:
            print('Hello World!')
        return python_dir.glob('libpypy3*.*')

    def to_lib(self, src):
        if False:
            print('Hello World!')
        return self.dest / 'lib' / src.name

    @classmethod
    def sources(cls, interpreter):
        if False:
            print('Hello World!')
        yield from super().sources(interpreter)
        if interpreter.system_prefix == '/usr':
            return
        host_lib = Path(interpreter.system_prefix) / 'lib'
        stdlib = Path(interpreter.system_stdlib)
        if host_lib.exists() and host_lib.is_dir():
            for path in host_lib.iterdir():
                if stdlib == path:
                    continue
                yield PathRefToDest(path, dest=cls.to_lib)

class Pypy3Windows(PyPy3, WindowsSupports):
    """PyPy 3 on Windows."""

    @property
    def less_v37(self):
        if False:
            while True:
                i = 10
        return self.interpreter.version_info.minor < 7

    @classmethod
    def _shared_libs(cls, python_dir):
        if False:
            print('Hello World!')
        for pattern in ['libpypy*.dll', 'libffi*.dll']:
            srcs = python_dir.glob(pattern)
            yield from srcs
__all__ = ['PyPy3', 'PyPy3Posix', 'Pypy3Windows']