import salt.utils.timed_subprocess as timed_subprocess
from tests.support.unit import TestCase

class TestTimedSubprocess(TestCase):

    def test_timedproc_with_shell_true_and_list_args(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This test confirms the fix for the regression introduced in 1f7d50d.\n        The TimedProc dunder init would result in a traceback if the args were\n        passed as a list and shell=True was set.\n        '
        p = timed_subprocess.TimedProc(['echo', 'foo'], shell=True)
        del p