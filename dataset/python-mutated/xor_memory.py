"""
xor-memory command test module
"""
from tests.utils import GefUnitTestGeneric, gdb_run_cmd, gdb_start_silent_cmd

class XorMemoryCommand(GefUnitTestGeneric):
    """`xor-memory` command test module"""

    def test_cmd_xor_memory_display(self):
        if False:
            return 10
        cmd = 'xor-memory display $sp 0x10 0x41'
        self.assertFailIfInactiveSession(gdb_run_cmd(cmd))
        res = gdb_start_silent_cmd(cmd)
        self.assertNoException(res)
        self.assertIn('Original block', res)
        self.assertIn('XOR-ed block', res)

    def test_cmd_xor_memory_patch(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = 'xor-memory patch $sp 0x10 0x41'
        res = gdb_start_silent_cmd(cmd)
        self.assertNoException(res)
        self.assertIn('Patching XOR-ing ', res)