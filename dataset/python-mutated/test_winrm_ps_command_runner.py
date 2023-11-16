from __future__ import absolute_import
import mock
from st2common.runners.base import ActionRunner
from st2tests.base import RunnerTestCase
from winrm_runner import winrm_ps_command_runner
from winrm_runner.winrm_base import WinRmBaseRunner

class WinRmPsCommandRunnerTestCase(RunnerTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(WinRmPsCommandRunnerTestCase, self).setUpClass()
        self._runner = winrm_ps_command_runner.get_runner()

    def test_init(self):
        if False:
            while True:
                i = 10
        runner = winrm_ps_command_runner.WinRmPsCommandRunner('abcdef')
        self.assertIsInstance(runner, WinRmBaseRunner)
        self.assertIsInstance(runner, ActionRunner)
        self.assertEqual(runner.runner_id, 'abcdef')

    @mock.patch('winrm_runner.winrm_ps_command_runner.WinRmPsCommandRunner.run_ps')
    def test_run(self, mock_run_ps):
        if False:
            while True:
                i = 10
        mock_run_ps.return_value = 'expected'
        self._runner.runner_parameters = {'cmd': 'Get-ADUser stanley'}
        result = self._runner.run({})
        self.assertEqual(result, 'expected')
        mock_run_ps.assert_called_with('Get-ADUser stanley')