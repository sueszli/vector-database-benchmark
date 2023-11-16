"""
mac_utils tests
"""
import os
import plistlib
import subprocess
import xml.parsers.expat
import pytest
import salt.modules.cmdmod as cmd
import salt.utils.mac_utils as mac_utils
import salt.utils.platform
from salt.exceptions import CommandExecutionError, SaltInvocationError
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import MagicMock, MockTimedProc, mock_open, patch
from tests.support.unit import TestCase

@pytest.mark.skip_unless_on_darwin
class MacUtilsTestCase(TestCase, LoaderModuleMockMixin):
    """
    test mac_utils salt utility
    """

    def setup_loader_modules(self):
        if False:
            for i in range(10):
                print('nop')
        return {mac_utils: {}}

    def test_execute_return_success_not_supported(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test execute_return_success function\n        command not supported\n        '
        mock_cmd = MagicMock(return_value={'retcode': 0, 'stdout': 'not supported', 'stderr': 'error'})
        with patch.object(mac_utils, '_run_all', mock_cmd):
            self.assertRaises(CommandExecutionError, mac_utils.execute_return_success, 'dir c:\\')

    def test_execute_return_success_command_failed(self):
        if False:
            i = 10
            return i + 15
        '\n        test execute_return_success function\n        command failed\n        '
        mock_cmd = MagicMock(return_value={'retcode': 1, 'stdout': 'spongebob', 'stderr': 'error'})
        with patch.object(mac_utils, '_run_all', mock_cmd):
            self.assertRaises(CommandExecutionError, mac_utils.execute_return_success, 'dir c:\\')

    def test_execute_return_success_command_succeeded(self):
        if False:
            return 10
        '\n        test execute_return_success function\n        command succeeded\n        '
        mock_cmd = MagicMock(return_value={'retcode': 0, 'stdout': 'spongebob'})
        with patch.object(mac_utils, '_run_all', mock_cmd):
            ret = mac_utils.execute_return_success('dir c:\\')
            self.assertEqual(ret, True)

    def test_execute_return_result_command_failed(self):
        if False:
            return 10
        '\n        test execute_return_result function\n        command failed\n        '
        mock_cmd = MagicMock(return_value={'retcode': 1, 'stdout': 'spongebob', 'stderr': 'squarepants'})
        with patch.object(mac_utils, '_run_all', mock_cmd):
            self.assertRaises(CommandExecutionError, mac_utils.execute_return_result, 'dir c:\\')

    def test_execute_return_result_command_succeeded(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test execute_return_result function\n        command succeeded\n        '
        mock_cmd = MagicMock(return_value={'retcode': 0, 'stdout': 'spongebob'})
        with patch.object(mac_utils, '_run_all', mock_cmd):
            ret = mac_utils.execute_return_result('dir c:\\')
            self.assertEqual(ret, 'spongebob')

    def test_parse_return_space(self):
        if False:
            return 10
        '\n        test parse_return function\n        space after colon\n        '
        self.assertEqual(mac_utils.parse_return('spongebob: squarepants'), 'squarepants')

    def test_parse_return_new_line(self):
        if False:
            return 10
        '\n        test parse_return function\n        new line after colon\n        '
        self.assertEqual(mac_utils.parse_return('spongebob:\nsquarepants'), 'squarepants')

    def test_parse_return_no_delimiter(self):
        if False:
            return 10
        '\n        test parse_return function\n        no delimiter\n        '
        self.assertEqual(mac_utils.parse_return('squarepants'), 'squarepants')

    def test_validate_enabled_on(self):
        if False:
            print('Hello World!')
        '\n        test validate_enabled function\n        test on\n        '
        self.assertEqual(mac_utils.validate_enabled('On'), 'on')

    def test_validate_enabled_off(self):
        if False:
            return 10
        '\n        test validate_enabled function\n        test off\n        '
        self.assertEqual(mac_utils.validate_enabled('Off'), 'off')

    def test_validate_enabled_bad_string(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test validate_enabled function\n        test bad string\n        '
        self.assertRaises(SaltInvocationError, mac_utils.validate_enabled, 'bad string')

    def test_validate_enabled_non_zero(self):
        if False:
            print('Hello World!')
        '\n        test validate_enabled function\n        test non zero\n        '
        for x in range(1, 179, 3):
            self.assertEqual(mac_utils.validate_enabled(x), 'on')

    def test_validate_enabled_0(self):
        if False:
            i = 10
            return i + 15
        '\n        test validate_enabled function\n        test 0\n        '
        self.assertEqual(mac_utils.validate_enabled(0), 'off')

    def test_validate_enabled_true(self):
        if False:
            while True:
                i = 10
        '\n        test validate_enabled function\n        test True\n        '
        self.assertEqual(mac_utils.validate_enabled(True), 'on')

    def test_validate_enabled_false(self):
        if False:
            print('Hello World!')
        '\n        test validate_enabled function\n        test False\n        '
        self.assertEqual(mac_utils.validate_enabled(False), 'off')

    def test_launchctl(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test launchctl function\n        '
        mock_cmd = MagicMock(return_value={'retcode': 0, 'stdout': 'success', 'stderr': 'none'})
        with patch('salt.utils.mac_utils.__salt__', {'cmd.run_all': mock_cmd}):
            ret = mac_utils.launchctl('enable', 'org.salt.minion')
            self.assertEqual(ret, True)

    def test_launchctl_return_stdout(self):
        if False:
            while True:
                i = 10
        '\n        test launchctl function and return stdout\n        '
        mock_cmd = MagicMock(return_value={'retcode': 0, 'stdout': 'success', 'stderr': 'none'})
        with patch('salt.utils.mac_utils.__salt__', {'cmd.run_all': mock_cmd}):
            ret = mac_utils.launchctl('enable', 'org.salt.minion', return_stdout=True)
            self.assertEqual(ret, 'success')

    def test_launchctl_error(self):
        if False:
            print('Hello World!')
        '\n        test launchctl function returning an error\n        '
        mock_cmd = MagicMock(return_value={'retcode': 1, 'stdout': 'failure', 'stderr': 'test failure'})
        error = 'Failed to enable service:\nstdout: failure\nstderr: test failure\nretcode: 1'
        with patch('salt.utils.mac_utils.__salt__', {'cmd.run_all': mock_cmd}):
            try:
                mac_utils.launchctl('enable', 'org.salt.minion')
            except CommandExecutionError as exc:
                self.assertEqual(exc.message, error)

    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    def test_available_services_result(self, mock_exists, mock_os_walk):
        if False:
            while True:
                i = 10
        '\n        test available_services results are properly formed dicts.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_exists.return_value = True
        plists = [{'Label': 'com.apple.lla1'}]
        ret = _run_available_services(plists)
        file_path = os.sep + os.path.join('Library', 'LaunchAgents', 'com.apple.lla1.plist')
        if salt.utils.platform.is_windows():
            file_path = 'c:' + file_path
        expected = {'com.apple.lla1': {'file_name': 'com.apple.lla1.plist', 'file_path': file_path, 'plist': plists[0]}}
        self.assertEqual(ret, expected)

    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    def test_available_services_dirs(self, mock_isdir, mock_listdir, mock_exists, mock_os_walk):
        if False:
            i = 10
            return i + 15
        '\n        test available_services checks all of the expected dirs.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist'], '/Library/LaunchDaemons': ['com.apple.lld1.plist'], '/System/Library/LaunchAgents': ['com.apple.slla1.plist'], '/System/Library/LaunchDaemons': ['com.apple.slld1.plist'], '/Users/saltymcsaltface/Library/LaunchAgents': ['com.apple.uslla1.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_listdir.return_value = ['saltymcsaltface']
        mock_isdir.return_value = True
        mock_exists.return_value = True
        plists = [{'Label': 'com.apple.lla1'}, {'Label': 'com.apple.lld1'}, {'Label': 'com.apple.slla1'}, {'Label': 'com.apple.slld1'}, {'Label': 'com.apple.uslla1'}]
        ret = _run_available_services(plists)
        self.assertEqual(len(ret), 5)

    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    @patch('plistlib.load')
    def test_available_services_broken_symlink(self, mock_read_plist, mock_exists, mock_os_walk):
        if False:
            print('Hello World!')
        '\n        test available_services when it encounters a broken symlink.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist', 'com.apple.lla2.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_exists.side_effect = [True, False]
        plists = [{'Label': 'com.apple.lla1'}]
        ret = _run_available_services(plists)
        file_path = os.sep + os.path.join('Library', 'LaunchAgents', 'com.apple.lla1.plist')
        if salt.utils.platform.is_windows():
            file_path = 'c:' + file_path
        expected = {'com.apple.lla1': {'file_name': 'com.apple.lla1.plist', 'file_path': file_path, 'plist': plists[0]}}
        self.assertEqual(ret, expected)

    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    @patch('salt.utils.mac_utils.__salt__')
    def test_available_services_binary_plist(self, mock_run, mock_exists, mock_os_walk):
        if False:
            print('Hello World!')
        '\n        test available_services handles binary plist files.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_exists.return_value = True
        plists = [{'Label': 'com.apple.lla1'}]
        file_path = os.sep + os.path.join('Library', 'LaunchAgents', 'com.apple.lla1.plist')
        if salt.utils.platform.is_windows():
            file_path = 'c:' + file_path
        ret = _run_available_services(plists)
        expected = {'com.apple.lla1': {'file_name': 'com.apple.lla1.plist', 'file_path': file_path, 'plist': plists[0]}}
        self.assertEqual(ret, expected)

    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    def test_available_services_invalid_file(self, mock_exists, mock_os_walk):
        if False:
            while True:
                i = 10
        '\n        test available_services excludes invalid files.\n        The py3 plistlib raises an InvalidFileException when a plist\n        file cannot be parsed.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_exists.return_value = True
        plists = [{'Label': 'com.apple.lla1'}]
        mock_load = MagicMock()
        mock_load.side_effect = plistlib.InvalidFileException
        with patch('salt.utils.files.fopen', mock_open()):
            with patch('plistlib.load', mock_load):
                ret = mac_utils._available_services()
        self.assertEqual(len(ret), 0)

    @patch('salt.utils.mac_utils.__salt__')
    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    def test_available_services_expat_error(self, mock_exists, mock_os_walk, mock_run):
        if False:
            for i in range(10):
                print('nop')
        '\n        test available_services excludes files with expat errors.\n\n        Poorly formed XML will raise an ExpatError on py2. It will\n        also be raised by some almost-correct XML on py3.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_exists.return_value = True
        file_path = os.sep + os.path.join('Library', 'LaunchAgents', 'com.apple.lla1.plist')
        if salt.utils.platform.is_windows():
            file_path = 'c:' + file_path
        mock_load = MagicMock()
        mock_load.side_effect = xml.parsers.expat.ExpatError
        with patch('salt.utils.files.fopen', mock_open()):
            with patch('plistlib.load', mock_load):
                ret = mac_utils._available_services()
        self.assertEqual(len(ret), 0)

    @patch('salt.utils.mac_utils.__salt__')
    @patch('salt.utils.path.os_walk')
    @patch('os.path.exists')
    def test_available_services_value_error(self, mock_exists, mock_os_walk, mock_run):
        if False:
            while True:
                i = 10
        '\n        test available_services excludes files with ValueErrors.\n        '
        results = {'/Library/LaunchAgents': ['com.apple.lla1.plist']}
        mock_os_walk.side_effect = _get_walk_side_effects(results)
        mock_exists.return_value = True
        file_path = os.sep + os.path.join('Library', 'LaunchAgents', 'com.apple.lla1.plist')
        if salt.utils.platform.is_windows():
            file_path = 'c:' + file_path
        mock_load = MagicMock()
        mock_load.side_effect = ValueError
        with patch('salt.utils.files.fopen', mock_open()):
            with patch('plistlib.load', mock_load):
                ret = mac_utils._available_services()
        self.assertEqual(len(ret), 0)

    def test_bootout_retcode_36_success(self):
        if False:
            print('Hello World!')
        '\n        Make sure that if we run a `launchctl bootout` cmd and it returns\n        36 that we treat it as a success.\n        '
        proc = MagicMock(return_value=MockTimedProc(stdout=None, stderr=None, returncode=36))
        with patch('salt.utils.timed_subprocess.TimedProc', proc):
            with patch('salt.utils.mac_utils.__salt__', {'cmd.run_all': cmd._run_all_quiet}):
                ret = mac_utils.launchctl('bootout', 'org.salt.minion')
        self.assertEqual(ret, True)

    def test_bootout_retcode_99_fail(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure that if we run a `launchctl bootout` cmd and it returns\n        something other than 0 or 36 that we treat it as a fail.\n        '
        error = 'Failed to bootout service:\nstdout: failure\nstderr: test failure\nretcode: 99'
        proc = MagicMock(return_value=MockTimedProc(stdout=b'failure', stderr=b'test failure', returncode=99))
        with patch('salt.utils.timed_subprocess.TimedProc', proc):
            with patch('salt.utils.mac_utils.__salt__', {'cmd.run_all': cmd._run_all_quiet}):
                try:
                    mac_utils.launchctl('bootout', 'org.salt.minion')
                except CommandExecutionError as exc:
                    self.assertEqual(exc.message, error)

    def test_not_bootout_retcode_36_fail(self):
        if False:
            i = 10
            return i + 15
        '\n        Make sure that if we get a retcode 36 on non bootout cmds\n        that we still get a failure.\n        '
        error = 'Failed to bootstrap service:\nstdout: failure\nstderr: test failure\nretcode: 36'
        proc = MagicMock(return_value=MockTimedProc(stdout=b'failure', stderr=b'test failure', returncode=36))
        with patch('salt.utils.timed_subprocess.TimedProc', proc):
            with patch('salt.utils.mac_utils.__salt__', {'cmd.run_all': cmd._run_all_quiet}):
                try:
                    mac_utils.launchctl('bootstrap', 'org.salt.minion')
                except CommandExecutionError as exc:
                    self.assertEqual(exc.message, error)

    def test_git_is_stub(self):
        if False:
            while True:
                i = 10
        mock_check_call = MagicMock(side_effect=subprocess.CalledProcessError(cmd='', returncode=2))
        with patch('salt.utils.mac_utils.subprocess.check_call', mock_check_call):
            self.assertEqual(mac_utils.git_is_stub(), True)

    @patch('salt.utils.mac_utils.subprocess.check_call')
    def test_git_is_not_stub(self, mock_check_call):
        if False:
            return 10
        self.assertEqual(mac_utils.git_is_stub(), False)

def _get_walk_side_effects(results):
    if False:
        print('Hello World!')
    '\n    Data generation helper function for service tests.\n    '

    def walk_side_effect(*args, **kwargs):
        if False:
            print('Hello World!')
        return [(args[0], [], results.get(args[0], []))]
    return walk_side_effect

def _run_available_services(plists):
    if False:
        print('Hello World!')
    mock_load = MagicMock()
    mock_load.side_effect = plists
    with patch('salt.utils.files.fopen', mock_open()):
        with patch('plistlib.load', mock_load):
            ret = mac_utils._available_services()
    return ret