from __future__ import absolute_import
import mock
import os.path
from st2common.runners.base import ActionRunner
from st2tests.base import RunnerTestCase
from winrm_runner import winrm_ps_script_runner
from winrm_runner.winrm_base import WinRmBaseRunner
from .fixtures import FIXTURES_PATH
POWERSHELL_SCRIPT_PATH = os.path.join(FIXTURES_PATH, 'TestScript.ps1')

class WinRmPsScriptRunnerTestCase(RunnerTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(WinRmPsScriptRunnerTestCase, self).setUpClass()
        self._runner = winrm_ps_script_runner.get_runner()

    def test_init(self):
        if False:
            i = 10
            return i + 15
        runner = winrm_ps_script_runner.WinRmPsScriptRunner('abcdef')
        self.assertIsInstance(runner, WinRmBaseRunner)
        self.assertIsInstance(runner, ActionRunner)
        self.assertEqual(runner.runner_id, 'abcdef')

    @mock.patch('winrm_runner.winrm_ps_script_runner.WinRmPsScriptRunner._get_script_args')
    @mock.patch('winrm_runner.winrm_ps_script_runner.WinRmPsScriptRunner.run_ps')
    def test_run(self, mock_run_ps, mock_get_script_args):
        if False:
            i = 10
            return i + 15
        mock_run_ps.return_value = 'expected'
        pos_args = [1, 'abc']
        named_args = {'d': {'test': ['\r', True, 3]}}
        mock_get_script_args.return_value = (pos_args, named_args)
        self._runner.entry_point = POWERSHELL_SCRIPT_PATH
        self._runner.runner_parameters = {}
        self._runner._kwarg_op = '-'
        result = self._runner.run({})
        self.assertEqual(result, 'expected')
        mock_run_ps.assert_called_with('[CmdletBinding()]\nParam(\n  [bool]$p_bool,\n  [int]$p_integer,\n  [double]$p_number,\n  [string]$p_str,\n  [array]$p_array,\n  [hashtable]$p_obj,\n  [Parameter(Position=0)]\n  [string]$p_pos0,\n  [Parameter(Position=1)]\n  [string]$p_pos1\n)\n\n\nWrite-Output "p_bool = $p_bool"\nWrite-Output "p_integer = $p_integer"\nWrite-Output "p_number = $p_number"\nWrite-Output "p_str = $p_str"\nWrite-Output "p_array = $($p_array | ConvertTo-Json -Compress)"\nWrite-Output "p_obj = $($p_obj | ConvertTo-Json -Compress)"\nWrite-Output "p_pos0 = $p_pos0"\nWrite-Output "p_pos1 = $p_pos1"\n', '-d @{"test" = @("`r", $true, 3)} 1 "abc"')