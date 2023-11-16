"""
GDB function test module for ELF section convenience functions
"""
from tests.utils import _target, gdb_run_cmd, gdb_run_silent_cmd, gdb_start_silent_cmd, is_64b
from tests.utils import GefUnitTestGeneric

class ElfSectionGdbFunction(GefUnitTestGeneric):
    """GDB functions test module"""

    def test_func_base(self):
        if False:
            while True:
                i = 10
        '`$_base()` GDB function test'
        cmd = 'x/s $_base()'
        self.assertFailIfInactiveSession(gdb_run_cmd(cmd))
        res = gdb_start_silent_cmd(cmd)
        self.assertNoException(res)
        self.assertIn('\\177ELF', res)
        addr = res.splitlines()[-1].split()[0][:-1]
        cmd = 'x/s $_base("libc")'
        res = gdb_start_silent_cmd(cmd)
        self.assertNoException(res)
        self.assertIn('\\177ELF', res)
        addr2 = res.splitlines()[-1].split()[0][:-1]
        self.assertNotEqual(addr, addr2)

    def test_func_bss(self):
        if False:
            i = 10
            return i + 15
        '`$_bss()` GDB function test'
        cmd = 'deref $_bss()'
        target = _target('bss')
        self.assertFailIfInactiveSession(gdb_run_cmd(cmd, target=target))
        res = gdb_run_silent_cmd(cmd, target=target)
        self.assertNoException(res)
        self.assertIn('Hello world!', res)

    def test_func_got(self):
        if False:
            while True:
                i = 10
        '`$_got()` GDB function test'
        cmd = 'deref $_got()'
        target = _target('heap')
        self.assertFailIfInactiveSession(gdb_run_cmd(cmd, target=target))
        res = gdb_run_silent_cmd(cmd, target=target)
        self.assertNoException(res)
        self.assertIn('malloc', res)

    def test_func_heap(self):
        if False:
            i = 10
            return i + 15
        '`$_heap()` GDB function test'
        cmd = 'deref $_heap()'
        target = _target('heap')
        self.assertFailIfInactiveSession(gdb_run_cmd(cmd, target=target))
        res = gdb_run_silent_cmd(cmd, target=target)
        self.assertNoException(res)
        if is_64b():
            self.assertIn('+0x0048:', res)
        else:
            self.assertIn('+0x0024:', res)
        cmd = 'deref $_heap(0x10+0x10)'
        res = gdb_run_silent_cmd(cmd, target=target)
        self.assertNoException(res)
        if is_64b():
            self.assertIn('+0x0048:', res)
        else:
            self.assertIn('+0x0024:', res)

    def test_func_stack(self):
        if False:
            for i in range(10):
                print('nop')
        '`$_stack()` GDB function test'
        cmd = 'deref $_stack()'
        self.assertFailIfInactiveSession(gdb_run_cmd(cmd))
        res = gdb_start_silent_cmd(cmd)
        self.assertNoException(res)
        if is_64b():
            self.assertRegex(res, '\\+0x0*20: *0x0000000000000000\\n')
        else:
            self.assertRegex(res, '\\+0x0.*20: *0x00000000\\n')