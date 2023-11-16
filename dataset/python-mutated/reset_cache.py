"""
`reset-cache` command test module
"""
from tests.utils import GefUnitTestGeneric, gdb_start_silent_cmd

class ResetCacheCommand(GefUnitTestGeneric):
    """`reset-cache` command test module"""

    def test_cmd_reset_cache(self):
        if False:
            print('Hello World!')
        res = gdb_start_silent_cmd('reset-cache')
        self.assertNoException(res)