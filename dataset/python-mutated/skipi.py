"""
`skipi` command test module
"""
import pytest
from tests.utils import ARCH, GefUnitTestGeneric, _target, findlines, gdb_run_cmd, gdb_run_silent_cmd, gdb_start_silent_cmd

class SkipiCommand(GefUnitTestGeneric):
    """`skipi` command test module"""
    cmd = 'skipi'

    def test_cmd_nop_inactive(self):
        if False:
            print('Hello World!')
        res = gdb_run_cmd(f'{self.cmd}')
        self.assertFailIfInactiveSession(res)

    @pytest.mark.skipif(ARCH not in ('i686', 'x86_64'), reason=f'Skipped for {ARCH}')
    def test_cmd_skipi_no_arg(self):
        if False:
            while True:
                i = 10
        res = gdb_start_silent_cmd('pi gef.memory.write(gef.arch.pc, p32(0x9090feeb))', after=(self.cmd, 'pi print(gef.memory.read(gef.arch.pc, 2))'))
        self.assertNoException(res)
        self.assertIn('\\x90\\x90', res)

    @pytest.mark.skipif(ARCH not in ('i686', 'x86_64'), reason=f'Skipped for {ARCH}')
    def test_cmd_skipi_skip_two_instructions(self):
        if False:
            i = 10
            return i + 15
        res = gdb_start_silent_cmd('pi gef.memory.write(gef.arch.pc, p64(0x90909090feebfeeb))', after=(f'{self.cmd} --n 2', 'pi print(gef.memory.read(gef.arch.pc, 4))'))
        self.assertNoException(res)
        self.assertIn('\\x90\\x90\\x90\\x90', res)

    @pytest.mark.skipif(ARCH not in ('i686', 'x86_64'), reason=f'Skipped for {ARCH}')
    def test_cmd_skipi_two_instructions_from_location(self):
        if False:
            i = 10
            return i + 15
        res = gdb_start_silent_cmd('pi gef.memory.write(gef.arch.pc, p64(0x9090feebfeebfeeb))', after=(f'{self.cmd} $pc+2 --n 2', 'pi print(gef.memory.read(gef.arch.pc, 2))'))
        self.assertNoException(res)
        self.assertIn('\\x90\\x90', res)