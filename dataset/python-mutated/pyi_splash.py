"""
This module connects to the bootloader to send messages to the splash screen.

It is intended to act as a RPC interface for the functions provided by the bootloader, such as displaying text or
closing. This makes the users python program independent of how the communication with the bootloader is implemented,
since a consistent API is provided.

To connect to the bootloader, it connects to a local tcp socket whose port is passed through the environment variable
'_PYIBoot_SPLASH'. The bootloader creates a server socket and accepts every connection request. Since the os-module,
which is needed to request the environment variable, is not available at boot time, the module does not establish the
connection until initialization.

The protocol by which the Python interpreter communicates with the bootloader is implemented in this module.

This module does not support reloads while the splash screen is displayed, i.e. it cannot be reloaded (such as by
importlib.reload), because the splash screen closes automatically when the connection to this instance of the module
is lost.
"""
import atexit
import os
import _socket
__all__ = ['CLOSE_CONNECTION', 'FLUSH_CHARACTER', 'is_alive', 'close', 'update_text']
try:
    import logging as _logging
except ImportError:
    _logging = None
try:
    from functools import update_wrapper
except ImportError:
    update_wrapper = None

def _log(level, msg, *args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Conditional wrapper around logging module. If the user excluded logging from the imports or it was not imported,\n    this function should handle it and avoid using the logger.\n    '
    if _logging:
        logger = _logging.getLogger(__name__)
        logger.log(level, msg, *args, **kwargs)
CLOSE_CONNECTION = b'\x04'
FLUSH_CHARACTER = b'\r'
_initialized = False
_ipc_socket_closed = True
_ipc_socket = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)

def _initialize():
    if False:
        print('Hello World!')
    '\n    Initialize this module\n\n    :return:\n    '
    global _initialized, _ipc_socket, _ipc_socket_closed
    try:
        _ipc_socket.connect(('localhost', _ipc_port))
        _ipc_socket_closed = False
        _initialized = True
        _log(20, 'A connection to the splash screen was successfully established.')
    except OSError as err:
        raise ConnectionError('Unable to connect to the tcp server socket on port %d' % _ipc_port) from err
try:
    _ipc_port = int(os.environ['_PYIBoot_SPLASH'])
    del os.environ['_PYIBoot_SPLASH']
    _initialize()
except (KeyError, ValueError) as _err:
    _log(30, 'The environment does not allow connecting to the splash screen. Are the splash resources attached to the bootloader or did an error occur?', exc_info=_err)
except ConnectionError as _err:
    _log(40, 'Cannot connect to the bootloaders ipc server socket', exc_info=_err)

def _check_connection(func):
    if False:
        i = 10
        return i + 15
    '\n    Utility decorator for checking whether the function should be executed.\n\n    The wrapped function may raise a ConnectionError if the module was not initialized correctly.\n    '

    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Executes the wrapped function if the environment allows it.\n\n        That is, if the connection to to bootloader has not been closed and the module is initialized.\n\n        :raises RuntimeError: if the module was not initialized correctly.\n        '
        if _initialized and _ipc_socket_closed:
            _log(20, 'The module has been disabled, so the use of the splash screen is no longer supported.')
            return
        elif not _initialized:
            raise RuntimeError('This module is not initialized; did it fail to load?')
        return func(*args, **kwargs)
    if update_wrapper:
        update_wrapper(wrapper, func)
    return wrapper

@_check_connection
def _send_command(cmd, args=None):
    if False:
        return 10
    '\n    Send the command followed by args to the splash screen.\n\n    :param str cmd: The command to send. All command have to be defined as procedures in the tcl splash screen script.\n    :param list[str] args: All arguments to send to the receiving function\n    '
    if args is None:
        args = []
    full_cmd = '%s(%s)' % (cmd, ' '.join(args))
    try:
        _ipc_socket.sendall(full_cmd.encode('utf-8') + FLUSH_CHARACTER)
    except OSError as err:
        raise ConnectionError("Unable to send '%s' to the bootloader" % full_cmd) from err

def is_alive():
    if False:
        print('Hello World!')
    '\n    Indicates whether the module can be used.\n\n    Returns False if the module is either not initialized or was disabled by closing the splash screen. Otherwise,\n    the module should be usable.\n    '
    return _initialized and (not _ipc_socket_closed)

@_check_connection
def update_text(msg: str):
    if False:
        return 10
    '\n    Updates the text on the splash screen window.\n\n    :param str msg: the text to be displayed\n    :raises ConnectionError: If the OS fails to write to the socket.\n    :raises RuntimeError: If the module is not initialized.\n    '
    _send_command('update_text', [msg])

def close():
    if False:
        return 10
    '\n    Close the connection to the ipc tcp server socket.\n\n    This will close the splash screen and renders this module unusable. After this function is called, no connection\n    can be opened to the splash screen again and all functions in this module become unusable.\n    '
    global _ipc_socket_closed
    if _initialized and (not _ipc_socket_closed):
        _ipc_socket.sendall(CLOSE_CONNECTION)
        _ipc_socket.close()
        _ipc_socket_closed = True

@atexit.register
def _exit():
    if False:
        for i in range(10):
            print('nop')
    close()