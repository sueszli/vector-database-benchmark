from __future__ import absolute_import
import mock
from st2common.runners.base import ActionRunner
from st2tests.base import RunnerTestCase
from winrm_runner import winrm_command_runner
from winrm_runner.winrm_base import WinRmBaseRunner

class WinRmCommandRunnerTestCase(RunnerTestCase):

    def setUp(self):
        if False:
            return 10
        super(WinRmCommandRunnerTestCase, self).setUpClass()
        self._runner = winrm_command_runner.get_runner()

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        runner = winrm_command_runner.WinRmCommandRunner('abcdef')
        self.assertIsInstance(runner, WinRmBaseRunner)
        self.assertIsInstance(runner, ActionRunner)
        self.assertEqual(runner.runner_id, 'abcdef')

    @mock.patch('winrm_runner.winrm_command_runner.WinRmCommandRunner.run_cmd')
    def test_run(self, mock_run_cmd):
        if False:
            i = 10
            return i + 15
        mock_run_cmd.return_value = 'expected'
        self._runner.runner_parameters = {'cmd': 'ipconfig /all'}
        result = self._runner.run({})
        self.assertEqual(result, 'expected')
        mock_run_cmd.assert_called_with('ipconfig /all')