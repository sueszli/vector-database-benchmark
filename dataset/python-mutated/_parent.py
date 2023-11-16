import os
from pathlib import Path
from marshal import loads, dumps
from base64 import b64encode, b64decode
import functools
import subprocess
import sys
from PyInstaller import compat
from PyInstaller import log as logging
logger = logging.getLogger(__name__)
if os.name == 'nt':
    import msvcrt
    import ctypes
    import ctypes.wintypes

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [('nLength', ctypes.wintypes.DWORD), ('lpSecurityDescriptor', ctypes.wintypes.LPVOID), ('bInheritHandle', ctypes.wintypes.BOOL)]
    HANDLE_FLAG_INHERIT = 1
    LPSECURITY_ATTRIBUTES = ctypes.POINTER(SECURITY_ATTRIBUTES)
    CreatePipe = ctypes.windll.kernel32.CreatePipe
    CreatePipe.argtypes = [ctypes.POINTER(ctypes.wintypes.HANDLE), ctypes.POINTER(ctypes.wintypes.HANDLE), LPSECURITY_ATTRIBUTES, ctypes.wintypes.DWORD]
    CreatePipe.restype = ctypes.wintypes.BOOL
    CloseHandle = ctypes.windll.kernel32.CloseHandle
    CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    CloseHandle.restype = ctypes.wintypes.BOOL
CHILD_PY = Path(__file__).with_name('_child.py')

def create_pipe(read_handle_inheritable, write_handle_inheritable):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a one-way pipe for sending data to child processes.\n\n    Args:\n        read_handle_inheritable:\n            A boolean flag indicating whether the handle corresponding to the read end-point of the pipe should be\n            marked as inheritable by subprocesses.\n        write_handle_inheritable:\n            A boolean flag indicating whether the handle corresponding to the write end-point of the pipe should be\n            marked as inheritable by subprocesses.\n\n    Returns:\n        A read/write pair of file descriptors (which are just integers) on posix or system file handles on Windows.\n\n    The pipe may be used either by this process or subprocesses of this process but not globally.\n    '
    return _create_pipe_impl(read_handle_inheritable, write_handle_inheritable)

def close_pipe_endpoint(pipe_handle):
    if False:
        print('Hello World!')
    '\n    Close the file descriptor (posix) or handle (Windows) belonging to a pipe.\n    '
    return _close_pipe_endpoint_impl(pipe_handle)
if os.name == 'nt':

    def _create_pipe_impl(read_handle_inheritable, write_handle_inheritable):
        if False:
            while True:
                i = 10
        read_handle = ctypes.wintypes.HANDLE()
        write_handle = ctypes.wintypes.HANDLE()
        security_attributes = SECURITY_ATTRIBUTES()
        security_attributes.nLength = ctypes.sizeof(security_attributes)
        security_attributes.bInheritHandle = True
        security_attributes.lpSecurityDescriptor = None
        succeeded = CreatePipe(ctypes.byref(read_handle), ctypes.byref(write_handle), ctypes.byref(security_attributes), 0)
        if not succeeded:
            raise ctypes.WinError()
        os.set_handle_inheritable(read_handle.value, read_handle_inheritable)
        os.set_handle_inheritable(write_handle.value, write_handle_inheritable)
        return (read_handle.value, write_handle.value)

    def _close_pipe_endpoint_impl(pipe_handle):
        if False:
            while True:
                i = 10
        succeeded = CloseHandle(pipe_handle)
        if not succeeded:
            raise ctypes.WinError()
else:

    def _create_pipe_impl(read_fd_inheritable, write_fd_inheritable):
        if False:
            return 10
        (read_fd, write_fd) = os.pipe()
        os.set_inheritable(read_fd, read_fd_inheritable)
        os.set_inheritable(write_fd, write_fd_inheritable)
        return (read_fd, write_fd)

    def _close_pipe_endpoint_impl(pipe_fd):
        if False:
            i = 10
            return i + 15
        os.close(pipe_fd)

def child(read_from_parent: int, write_to_parent: int):
    if False:
        for i in range(10):
            print('nop')
    '\n    Spawn a Python subprocess sending it the two file descriptors it needs to talk back to this parent process.\n    '
    if os.name != 'nt':
        extra_kwargs = {'env': _subprocess_env(), 'close_fds': False}
    else:
        extra_kwargs = {'env': _subprocess_env(), 'close_fds': True, 'startupinfo': subprocess.STARTUPINFO(lpAttributeList={'handle_list': [read_from_parent, write_to_parent]})}
    (cmd, options) = compat.__wrap_python([str(CHILD_PY), str(read_from_parent), str(write_to_parent)], extra_kwargs)
    return subprocess.Popen(cmd, **options)

def _subprocess_env():
    if False:
        for i in range(10):
            print('nop')
    '\n    Define the environment variables to be readable in a child process.\n    '
    from PyInstaller.config import CONF
    python_path = CONF['pathex']
    if 'PYTHONPATH' in os.environ:
        python_path = python_path + [os.environ['PYTHONPATH']]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(python_path)
    return env

class SubprocessDiedError(RuntimeError):
    pass

class Python:
    """
    Start and connect to a separate Python subprocess.

    This is the lowest level of public API provided by this module. The advantage of using this class directly is
    that it allows multiple functions to be evaluated in a single subprocess, making it faster than multiple calls to
    :func:`call`.

    The ``strict_mode`` argument controls behavior when the child process fails to shut down; if strict mode is enabled,
    an error is raised, otherwise only warning is logged. If the value of ``strict_mode`` is ``None``, the value of
    ``PyInstaller.compat.strict_collect_mode`` is used (which in turn is controlled by the
    ``PYINSTALLER_STRICT_COLLECT_MODE`` environment variable.

    Examples:
        To call some predefined functions ``x = foo()``, ``y = bar("numpy")`` and ``z = bazz(some_flag=True)`` all using
        the same isolated subprocess use::

            with isolated.Python() as child:
                x = child.call(foo)
                y = child.call(bar, "numpy")
                z = child.call(bazz, some_flag=True)

    """

    def __init__(self, strict_mode=None):
        if False:
            return 10
        self._child = None
        self._strict_mode = strict_mode if strict_mode is not None else compat.strict_collect_mode

    def __enter__(self):
        if False:
            return 10
        (read_from_child, write_to_parent) = create_pipe(False, True)
        (read_from_parent, write_to_child) = create_pipe(True, False)
        self._child = child(read_from_parent, write_to_parent)
        close_pipe_endpoint(read_from_parent)
        close_pipe_endpoint(write_to_parent)
        del read_from_parent
        del write_to_parent
        if os.name == 'nt':
            self._write_handle = os.fdopen(msvcrt.open_osfhandle(write_to_child, 0), 'wb')
            self._read_handle = os.fdopen(msvcrt.open_osfhandle(read_from_child, 0), 'rb')
        else:
            self._write_handle = os.fdopen(write_to_child, 'wb')
            self._read_handle = os.fdopen(read_from_child, 'rb')
        self._send(sys.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        if exc_type and issubclass(exc_type, SubprocessDiedError):
            self._write_handle.close()
            self._read_handle.close()
            del self._read_handle, self._write_handle
            self._child = None
            return
        self._write_handle.write(b'\n')
        self._write_handle.flush()
        shutdown_error = False
        try:
            self._child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning('Timed out while waiting for the child process to exit!')
            shutdown_error = True
            self._child.kill()
            try:
                self._child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning('Timed out while waiting for the child process to be killed!')
        self._write_handle.close()
        self._read_handle.close()
        del self._read_handle, self._write_handle
        self._child = None
        if shutdown_error and self._strict_mode:
            raise RuntimeError('Timed out while waiting for the child process to exit!')

    def call(self, function, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Call a function in the child Python. Retrieve its return value. Usage of this method is identical to that\n        of the :func:`call` function.\n        '
        if self._child is None:
            raise RuntimeError("An isolated.Python object must be used in a 'with' clause.")
        self._send(function.__code__, function.__defaults__, function.__kwdefaults__, args, kwargs)
        try:
            (ok, output) = loads(b64decode(self._read_handle.readline()))
        except (EOFError, BrokenPipeError):
            raise SubprocessDiedError(f'Child process died calling {function.__name__}() with args={args} and kwargs={kwargs}. Its exit code was {self._child.wait()}.') from None
        if ok:
            return output
        raise RuntimeError(f'Child process call to {function.__name__}() failed with:\n' + output)

    def _send(self, *objects):
        if False:
            while True:
                i = 10
        for object in objects:
            self._write_handle.write(b64encode(dumps(object)))
            self._write_handle.write(b'\n')
        self._write_handle.flush()

def call(function, *args, **kwargs):
    if False:
        return 10
    '\n    Call a function with arguments in a separate child Python. Retrieve its return value.\n\n    Args:\n        function:\n            The function to send and invoke.\n        *args:\n        **kwargs:\n            Positional and keyword arguments to send to the function. These must be simple builtin types - not custom\n            classes.\n    Returns:\n        The return value of the function. Again, these must be basic types serialisable by :func:`marshal.dumps`.\n    Raises:\n        RuntimeError:\n            Any exception which happens inside an isolated process is caught and reraised in the parent process.\n\n    To use, define a function which returns the information you\'re looking for. Any imports it requires must happen in\n    the body of the function. For example, to safely check the output of ``matplotlib.get_data_path()`` use::\n\n        # Define a function to be ran in isolation.\n        def get_matplotlib_data_path():\n            import matplotlib\n            return matplotlib.get_data_path()\n\n        # Call it with isolated.call().\n        get_matplotlib_data_path = isolated.call(matplotlib_data_path)\n\n    For single use functions taking no arguments like the above you can abuse the decorator syntax slightly to define\n    and execute a function in one go. ::\n\n        >>> @isolated.call\n        ... def matplotlib_data_dir():\n        ...     import matplotlib\n        ...     return matplotlib.get_data_path()\n        >>> matplotlib_data_dir\n        \'/home/brenainn/.pyenv/versions/3.9.6/lib/python3.9/site-packages/matplotlib/mpl-data\'\n\n    Functions may take positional and keyword arguments and return most generic Python data types. ::\n\n        >>> def echo_parameters(*args, **kwargs):\n        ...     return args, kwargs\n        >>> isolated.call(echo_parameters, 1, 2, 3)\n        (1, 2, 3), {}\n        >>> isolated.call(echo_parameters, foo=["bar"])\n        (), {\'foo\': [\'bar\']}\n\n    Notes:\n        To make a function behave differently if it\'s isolated, check for the ``__isolated__`` global. ::\n\n            if globals().get("__isolated__", False):\n                # We\'re inside a child process.\n                ...\n            else:\n                # This is the master process.\n                ...\n\n    '
    with Python() as isolated:
        return isolated.call(function, *args, **kwargs)

def decorate(function):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorate a function so that it is always called in an isolated subprocess.\n\n    Examples:\n\n        To use, write a function then prepend ``@isolated.decorate``. ::\n\n            @isolated.decorate\n            def add_1(x):\n                \'\'\'Add 1 to ``x``, displaying the current process ID.\'\'\'\n                import os\n                print(f"Process {os.getpid()}: Adding 1 to {x}.")\n                return x + 1\n\n        The resultant ``add_1()`` function can now be called as you would a\n        normal function and it\'ll automatically use a subprocess.\n\n            >>> add_1(4)\n            Process 4920: Adding 1 to 4.\n            5\n            >>> add_1(13.2)\n            Process 4928: Adding 1 to 13.2.\n            14.2\n\n    '

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if False:
            print('Hello World!')
        return call(function, *args, **kwargs)
    return wrapped