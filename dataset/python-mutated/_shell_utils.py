"""
Helper functions for interacting with the shell, and consuming shell-style
parameters provided in config files.
"""
import os
import shlex
import subprocess
__all__ = ['WindowsParser', 'PosixParser', 'NativeParser']

class CommandLineParser:
    """
    An object that knows how to split and join command-line arguments.

    It must be true that ``argv == split(join(argv))`` for all ``argv``.
    The reverse neednt be true - `join(split(cmd))` may result in the addition
    or removal of unnecessary escaping.
    """

    @staticmethod
    def join(argv):
        if False:
            return 10
        ' Join a list of arguments into a command line string '
        raise NotImplementedError

    @staticmethod
    def split(cmd):
        if False:
            i = 10
            return i + 15
        ' Split a command line string into a list of arguments '
        raise NotImplementedError

class WindowsParser:
    """
    The parsing behavior used by `subprocess.call("string")` on Windows, which
    matches the Microsoft C/C++ runtime.

    Note that this is _not_ the behavior of cmd.
    """

    @staticmethod
    def join(argv):
        if False:
            print('Hello World!')
        return subprocess.list2cmdline(argv)

    @staticmethod
    def split(cmd):
        if False:
            while True:
                i = 10
        import ctypes
        try:
            ctypes.windll
        except AttributeError:
            raise NotImplementedError
        if not cmd:
            return []
        cmd = 'dummy ' + cmd
        CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
        CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
        nargs = ctypes.c_int()
        lpargs = CommandLineToArgvW(cmd, ctypes.byref(nargs))
        args = [lpargs[i] for i in range(nargs.value)]
        assert not ctypes.windll.kernel32.LocalFree(lpargs)
        assert args[0] == 'dummy'
        return args[1:]

class PosixParser:
    """
    The parsing behavior used by `subprocess.call("string", shell=True)` on Posix.
    """

    @staticmethod
    def join(argv):
        if False:
            return 10
        return ' '.join((shlex.quote(arg) for arg in argv))

    @staticmethod
    def split(cmd):
        if False:
            print('Hello World!')
        return shlex.split(cmd, posix=True)
if os.name == 'nt':
    NativeParser = WindowsParser
elif os.name == 'posix':
    NativeParser = PosixParser