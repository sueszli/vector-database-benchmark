import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from coalib import coala
from coala_utils.ContextManagers import prepare_file
from tests.TestUtilities import execute_coala, bear_test_module

@patch('coalib.coala_modes.mode_json')
class coalaDebugFlagTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.old_argv = sys.argv

    def pipReqIsInstalledMock(self):
        if False:
            i = 10
            return i + 15
        '\n        Prepare a patch for ``PipRequirement.is_installed`` method that\n        always returns ``True``, used for faking an installed ipdb.\n        '
        return patch('dependency_management.requirements.PipRequirement.PipRequirement.is_installed', lambda self: True)

    def pipReqIsNotInstalledMock(self):
        if False:
            print('Hello World!')
        '\n        Prepare a patch for ``PipRequirement.is_installed`` method that\n        always returns ``False``, used for faking a not installed ipdb.\n        '
        return patch('dependency_management.requirements.PipRequirement.PipRequirement.is_installed', lambda self: False)

    def ipdbMock(self):
        if False:
            return 10
        '\n        Prepare a mocked ``ipdb`` module with a mocked\n        ``launch_ipdb_on_exception`` function, which is used in\n        ``coala --debug`` mode to open and ``ipdb>`` prompt when unexpected\n        exceptions occur\n        '
        mock = MagicMock()

        def __exit__(self, *exc_info):
            if False:
                i = 10
                return i + 15
            '\n            Make mocked ``ipdb.launch_ipdb_on_exception()`` context just\n            reraise the exception.\n            '
            raise
        mock.launch_ipdb_on_exception.__enter__ = None
        mock.launch_ipdb_on_exception.__exit__ = __exit__
        return mock

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        sys.argv = self.old_argv

    def test_no_ipdb(self, mocked_mode_json):
        if False:
            print('Hello World!')
        mocked_mode_json.side_effect = None
        with bear_test_module(), self.pipReqIsNotInstalledMock():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                (status, stdout, stderr) = execute_coala(coala.main, 'coala', '--debug', '--json', '-c', os.devnull, '-f', filename, '-b', 'RaiseTestBear')
        assert status == 13
        assert not stdout
        assert '--debug flag requires ipdb.' in stderr

    def test_bear__init__raises(self, mocked_mode_json):
        if False:
            return 10
        mocked_mode_json.side_effect = None
        mocked_ipdb = self.ipdbMock()
        with bear_test_module(), self.pipReqIsInstalledMock():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with patch.dict('sys.modules', ipdb=mocked_ipdb):
                    with self.assertRaisesRegex(RuntimeError, "^The bear ErrorTestBear does not fulfill all requirements\\. 'I_do_not_exist' is not installed\\.$"):
                        execute_coala(coala.main, 'coala', '--debug', '-c', os.devnull, '-f', filename, '-b', 'ErrorTestBear')
        mocked_ipdb.launch_ipdb_on_exception.assert_called_once_with()

    def test_bear_run_raises(self, mocked_mode_json):
        if False:
            while True:
                i = 10
        mocked_mode_json.side_effect = None
        mocked_ipdb = self.ipdbMock()
        with bear_test_module(), self.pipReqIsInstalledMock():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with patch.dict('sys.modules', ipdb=mocked_ipdb):
                    with self.assertRaisesRegex(RuntimeError, "^That's all the RaiseTestBear can do\\.$"):
                        execute_coala(coala.main, 'coala', '--debug', '-c', os.devnull, '-f', filename, '-b', 'RaiseTestBear')
        mocked_ipdb.launch_ipdb_on_exception.assert_called_once_with()

    def test_coala_main_mode_json_launches_ipdb(self, mocked_mode_json):
        if False:
            return 10
        mocked_mode_json.side_effect = RuntimeError('Mocked mode_json fails.')
        mocked_ipdb = self.ipdbMock()
        with bear_test_module(), self.pipReqIsInstalledMock():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with patch.dict('sys.modules', ipdb=mocked_ipdb):
                    with self.assertRaisesRegex(RuntimeError, '^Mocked mode_json fails\\.$'):
                        execute_coala(coala.main, 'coala', '--debug', '--json', '-c', os.devnull, '-f', filename, '-b', 'RaiseTestBear')
        mocked_ipdb.launch_ipdb_on_exception.assert_called_once_with()