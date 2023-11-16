"""
The child process to be invoked by IsolatedPython().

This file is to be run directly with pipe handles for reading from and writing to the parent process as command line
arguments.

"""
import sys
import os
import types
from marshal import loads, dumps
from base64 import b64encode, b64decode
from traceback import format_exception
if os.name == 'nt':
    from msvcrt import open_osfhandle

    def _open(osf_handle, mode):
        if False:
            for i in range(10):
                print('nop')
        return open(open_osfhandle(osf_handle, 0), mode)
else:
    _open = open

def run_next_command(read_fh, write_fh):
    if False:
        return 10
    '\n    Listen to **read_fh** for the next function to run. Write the result to **write_fh**.\n    '
    first_line = read_fh.readline()
    if first_line == b'\n':
        return False
    code = loads(b64decode(first_line.strip()))
    _defaults = loads(b64decode(read_fh.readline().strip()))
    _kwdefaults = loads(b64decode(read_fh.readline().strip()))
    args = loads(b64decode(read_fh.readline().strip()))
    kwargs = loads(b64decode(read_fh.readline().strip()))
    try:
        GLOBALS = {'__builtins__': __builtins__, '__isolated__': True}
        function = types.FunctionType(code, GLOBALS)
        function.__defaults__ = _defaults
        function.__kwdefaults__ = _kwdefaults
        output = function(*args, **kwargs)
        marshalled = dumps((True, output))
    except BaseException as ex:
        tb_lines = format_exception(type(ex), ex, ex.__traceback__)
        if tb_lines[0] == 'Traceback (most recent call last):\n':
            tb_lines = tb_lines[1:]
        marshalled = dumps((False, ''.join(tb_lines).rstrip()))
    write_fh.write(b64encode(marshalled))
    write_fh.write(b'\n')
    write_fh.flush()
    return True
if __name__ == '__main__':
    (read_from_parent, write_to_parent) = map(int, sys.argv[1:])
    with _open(read_from_parent, 'rb') as read_fh:
        with _open(write_to_parent, 'wb') as write_fh:
            sys.path = loads(b64decode(read_fh.readline()))
            while run_next_command(read_fh, write_fh):
                pass