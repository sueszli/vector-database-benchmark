"""Helpers for running gdb related testing"""
import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
try:
    _confirm_gdb(need_ptrace_attach=False)
    _HAVE_GDB = True
    _gdb_info = collect_gdbinfo()
    _GDB_HAS_PY3 = _gdb_info.py_ver.startswith('3')
except Exception:
    _HAVE_GDB = False
    _GDB_HAS_PY3 = False
_msg = 'functioning gdb with correct ptrace permissions is required'
needs_gdb = unittest.skipUnless(_HAVE_GDB, _msg)
_msg = 'gdb with python 3 support needed'
needs_gdb_py3 = unittest.skipUnless(_GDB_HAS_PY3, _msg)
try:
    import pexpect
    _HAVE_PEXPECT = True
except ImportError:
    _HAVE_PEXPECT = False
_msg = 'pexpect module needed for test'
skip_unless_pexpect = unittest.skipUnless(_HAVE_PEXPECT, _msg)

class GdbMIDriver(object):
    """
    Driver class for the GDB machine interface:
    https://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI.html
    """

    def __init__(self, file_name, debug=False, timeout=120, init_cmds=None):
        if False:
            while True:
                i = 10
        if not _HAVE_PEXPECT:
            msg = 'This driver requires the pexpect module. This can be obtained via:\n\n$ conda install pexpect'
            raise RuntimeError(msg)
        if not _HAVE_GDB:
            msg = 'This driver requires a gdb binary. This can be obtained via the system package manager.'
            raise RuntimeError(msg)
        self._gdb_binary = config.GDB_BINARY
        self._python = sys.executable
        self._debug = debug
        self._file_name = file_name
        self._timeout = timeout
        self._init_cmds = init_cmds
        self._drive()

    def _drive(self):
        if False:
            while True:
                i = 10
        'This function sets up the caputured gdb instance'
        assert os.path.isfile(self._file_name)
        cmd = [self._gdb_binary, '--interpreter', 'mi']
        if self._init_cmds is not None:
            cmd += list(self._init_cmds)
        cmd += ['--args', self._python, self._file_name]
        self._captured = pexpect.spawn(' '.join(cmd))
        if self._debug:
            self._captured.logfile = sys.stdout.buffer

    def supports_python(self):
        if False:
            print('Hello World!')
        'Returns True if the underlying gdb implementation has python support\n           False otherwise'
        return 'python' in self.list_features()

    def supports_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the underlying gdb implementation has NumPy support\n           (and by extension Python support) False otherwise'
        if not self.supports_python():
            return False
        cmd = 'python from __future__ import print_function;import numpy; print(numpy)'
        self.interpreter_exec('console', cmd)
        return "module 'numpy' from" in self._captured.before.decode()

    def _captured_expect(self, expect):
        if False:
            while True:
                i = 10
        try:
            self._captured.expect(expect, timeout=self._timeout)
        except pexpect.exceptions.TIMEOUT as e:
            msg = f'Expected value did not arrive: {expect}.'
            raise ValueError(msg) from e

    def assert_output(self, expected):
        if False:
            print('Hello World!')
        'Asserts that the current output string contains the expected.'
        output = self._captured.after
        decoded = output.decode('utf-8')
        assert expected in decoded, f'decoded={decoded}\nexpected={expected})'

    def assert_regex_output(self, expected):
        if False:
            for i in range(10):
                print('nop')
        'Asserts that the current output string contains the expected\n        regex.'
        output = self._captured.after
        decoded = output.decode('utf-8')
        done_str = decoded.splitlines()[0]
        found = re.match(expected, done_str)
        assert found, f'decoded={decoded}\nexpected={expected})'

    def _run_command(self, command, expect=''):
        if False:
            i = 10
            return i + 15
        self._captured.sendline(command)
        self._captured_expect(expect)

    def run(self):
        if False:
            print('Hello World!')
        "gdb command ~= 'run'"
        self._run_command('-exec-run', expect='\\^running.*\\r\\n')

    def cont(self):
        if False:
            i = 10
            return i + 15
        "gdb command ~= 'continue'"
        self._run_command('-exec-continue', expect='\\^running.*\\r\\n')

    def quit(self):
        if False:
            i = 10
            return i + 15
        "gdb command ~= 'quit'"
        self._run_command('-gdb-exit', expect='-gdb-exit')
        self._captured.terminate()

    def next(self):
        if False:
            i = 10
            return i + 15
        "gdb command ~= 'next'"
        self._run_command('-exec-next', expect='\\*stopped,.*\\r\\n')

    def step(self):
        if False:
            while True:
                i = 10
        "gdb command ~= 'step'"
        self._run_command('-exec-step', expect='\\*stopped,.*\\r\\n')

    def set_breakpoint(self, line=None, symbol=None, condition=None):
        if False:
            print('Hello World!')
        "gdb command ~= 'break'"
        if line is not None and symbol is not None:
            raise ValueError('Can only supply one of line or symbol')
        bp = '-break-insert '
        if condition is not None:
            bp += f'-c "{condition}" '
        if line is not None:
            assert isinstance(line, int)
            bp += f'-f {self._file_name}:{line} '
        if symbol is not None:
            assert isinstance(symbol, str)
            bp += f'-f {symbol} '
        self._run_command(bp, expect='\\^done')

    def check_hit_breakpoint(self, number=None, line=None):
        if False:
            return 10
        'Checks that a breakpoint has been hit'
        self._captured_expect('\\*stopped,.*\\r\\n')
        self.assert_output('*stopped,reason="breakpoint-hit",')
        if number is not None:
            assert isinstance(number, int)
            self.assert_output(f'bkptno="{number}"')
        if line is not None:
            assert isinstance(line, int)
            self.assert_output(f'line="{line}"')

    def stack_list_arguments(self, print_values=1, low_frame=0, high_frame=0):
        if False:
            while True:
                i = 10
        "gdb command ~= 'info args'"
        for x in (print_values, low_frame, high_frame):
            assert isinstance(x, int) and x in (0, 1, 2)
        cmd = f'-stack-list-arguments {print_values} {low_frame} {high_frame}'
        self._run_command(cmd, expect='\\^done,.*\\r\\n')

    def stack_list_variables(self, print_values=1):
        if False:
            i = 10
            return i + 15
        "gdb command ~= 'info locals'"
        assert isinstance(print_values, int) and print_values in (0, 1, 2)
        cmd = f'-stack-list-variables {print_values}'
        self._run_command(cmd, expect='\\^done,.*\\r\\n')

    def interpreter_exec(self, interpreter=None, command=None):
        if False:
            return 10
        "gdb command ~= 'interpreter-exec'"
        if interpreter is None:
            raise ValueError('interpreter cannot be None')
        if command is None:
            raise ValueError('command cannot be None')
        cmd = f'-interpreter-exec {interpreter} "{command}"'
        self._run_command(cmd, expect='\\^(done|error).*\\r\\n')

    def _list_features_raw(self):
        if False:
            i = 10
            return i + 15
        cmd = '-list-features'
        self._run_command(cmd, expect='\\^done,.*\\r\\n')

    def list_features(self):
        if False:
            for i in range(10):
                print('nop')
        'No equivalent gdb command? Returns a list of supported gdb\n           features.\n        '
        self._list_features_raw()
        output = self._captured.after
        decoded = output.decode('utf-8')
        m = re.match('.*features=\\[(.*)\\].*', decoded)
        assert m is not None, 'No match found for features string'
        g = m.groups()
        assert len(g) == 1, 'Invalid number of match groups found'
        return g[0].replace('"', '').split(',')