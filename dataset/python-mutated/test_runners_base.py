from __future__ import absolute_import
from st2common.runners.base import get_runner
from st2common.exceptions.actionrunner import ActionRunnerCreateError
from st2tests.base import DbTestCase

class RunnersLoaderUtilsTestCase(DbTestCase):

    def test_get_runner_success(self):
        if False:
            while True:
                i = 10
        runner = get_runner('local-shell-cmd')
        self.assertTrue(runner)
        self.assertEqual(runner.__class__.__name__, 'LocalShellCommandRunner')

    def test_get_runner_failure_not_found(self):
        if False:
            i = 10
            return i + 15
        expected_msg = 'Failed to find runner invalid-name-not-found.*'
        self.assertRaisesRegexp(ActionRunnerCreateError, expected_msg, get_runner, 'invalid-name-not-found')