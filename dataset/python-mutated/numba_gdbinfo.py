"""Module for displaying information about Numba's gdb set up"""
from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
_fields = 'binary_loc, extension_loc, py_ver, np_ver, supported'
_gdb_info = namedtuple('_gdb_info', _fields)

class _GDBTestWrapper:
    """Wraps the gdb binary and has methods for checking what the gdb binary
    has support for (Python and NumPy)."""

    def __init__(self):
        if False:
            while True:
                i = 10
        gdb_binary = config.GDB_BINARY
        if gdb_binary is None:
            msg = f'No valid binary could be found for gdb named: {config.GDB_BINARY}'
            raise ValueError(msg)
        self._gdb_binary = gdb_binary

    def _run_cmd(self, cmd=()):
        if False:
            while True:
                i = 10
        gdb_call = [self.gdb_binary, '-q']
        for x in cmd:
            gdb_call.append('-ex')
            gdb_call.append(x)
        gdb_call.extend(['-ex', 'q'])
        return subprocess.run(gdb_call, capture_output=True, timeout=10, text=True)

    @property
    def gdb_binary(self):
        if False:
            print('Hello World!')
        return self._gdb_binary

    @classmethod
    def success(cls, status):
        if False:
            print('Hello World!')
        return status.returncode == 0

    def check_launch(self):
        if False:
            while True:
                i = 10
        'Checks that gdb will launch ok'
        return self._run_cmd()

    def check_python(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = 'python from __future__ import print_function; import sys; print(sys.version_info[:2])'
        return self._run_cmd((cmd,))

    def check_numpy(self):
        if False:
            print('Hello World!')
        cmd = 'python from __future__ import print_function; import types; import numpy; print(isinstance(numpy, types.ModuleType))'
        return self._run_cmd((cmd,))

    def check_numpy_version(self):
        if False:
            print('Hello World!')
        cmd = 'python from __future__ import print_function; import types; import numpy;print(numpy.__version__)'
        return self._run_cmd((cmd,))

def collect_gdbinfo():
    if False:
        i = 10
        return i + 15
    'Prints information to stdout about the gdb setup that Numba has found'
    gdb_state = None
    gdb_has_python = False
    gdb_has_numpy = False
    gdb_python_version = 'No Python support'
    gdb_python_numpy_version = 'No NumPy support'
    try:
        gdb_wrapper = _GDBTestWrapper()
        status = gdb_wrapper.check_launch()
        if not gdb_wrapper.success(status):
            msg = f"gdb at '{gdb_wrapper.gdb_binary}' does not appear to work.\nstdout: {status.stdout}\nstderr: {status.stderr}"
            raise ValueError(msg)
        gdb_state = gdb_wrapper.gdb_binary
    except Exception as e:
        gdb_state = f'Testing gdb binary failed. Reported Error: {e}'
    else:
        status = gdb_wrapper.check_python()
        if gdb_wrapper.success(status):
            version_match = re.match('\\((\\d+),\\s+(\\d+)\\)', status.stdout.strip())
            if version_match is not None:
                (pymajor, pyminor) = version_match.groups()
                gdb_python_version = f'{pymajor}.{pyminor}'
                gdb_has_python = True
                status = gdb_wrapper.check_numpy()
                if gdb_wrapper.success(status):
                    if 'Traceback' not in status.stderr.strip():
                        if status.stdout.strip() == 'True':
                            gdb_has_numpy = True
                            gdb_python_numpy_version = 'Unknown'
                            status = gdb_wrapper.check_numpy_version()
                            if gdb_wrapper.success(status):
                                if 'Traceback' not in status.stderr.strip():
                                    gdb_python_numpy_version = status.stdout.strip()
    if gdb_has_python:
        if gdb_has_numpy:
            print_ext_supported = 'Full (Python and NumPy supported)'
        else:
            print_ext_supported = 'Partial (Python only, no NumPy support)'
    else:
        print_ext_supported = 'None'
    print_ext_file = 'gdb_print_extension.py'
    print_ext_path = os.path.join(os.path.dirname(__file__), print_ext_file)
    return _gdb_info(gdb_state, print_ext_path, gdb_python_version, gdb_python_numpy_version, print_ext_supported)

def display_gdbinfo(sep_pos=45):
    if False:
        while True:
            i = 10
    'Displays the information collected by collect_gdbinfo.\n    '
    gdb_info = collect_gdbinfo()
    print('-' * 80)
    fmt = f'%-{sep_pos}s : %-s'
    print(fmt % ('Binary location', gdb_info.binary_loc))
    print(fmt % ('Print extension location', gdb_info.extension_loc))
    print(fmt % ('Python version', gdb_info.py_ver))
    print(fmt % ('NumPy version', gdb_info.np_ver))
    print(fmt % ('Numba printing extension support', gdb_info.supported))
    print('')
    print('To load the Numba gdb printing extension, execute the following from the gdb prompt:')
    print(f'\nsource {gdb_info.extension_loc}\n')
    print('-' * 80)
    warn = '\n    =============================================================\n    IMPORTANT: Before sharing you should remove any information\n    in the above that you wish to keep private e.g. paths.\n    =============================================================\n    '
    print(dedent(warn))
if __name__ == '__main__':
    display_gdbinfo()