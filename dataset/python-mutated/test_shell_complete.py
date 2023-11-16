"""Black-box tests for 'bzr shell-complete'."""
from bzrlib.tests import TestCaseWithTransport

class TestShellComplete(TestCaseWithTransport):

    def test_shell_complete(self):
        if False:
            print('Hello World!')
        self.run_bzr('shell-complete')