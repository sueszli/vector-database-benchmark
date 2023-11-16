"""Tests the public utility functions in path_watcher.py"""
import unittest
from unittest.mock import Mock, call, patch
import streamlit.watcher.path_watcher
from streamlit.watcher.path_watcher import NoOpPathWatcher, get_default_path_watcher_class, watch_dir, watch_file
from tests.testutil import patch_config_options

class FileWatcherTest(unittest.TestCase):

    def test_report_watchdog_availability_mac(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('streamlit.watcher.path_watcher.watchdog_available', new=False), patch('streamlit.env_util.IS_DARWIN', new=True), patch('click.secho') as mock_echo:
            streamlit.watcher.path_watcher.report_watchdog_availability()
        msg = '\n  $ xcode-select --install'
        calls = [call('  %s' % 'For better performance, install the Watchdog module:', fg='blue', bold=True), call('%s\n  $ pip install watchdog\n            ' % msg)]
        mock_echo.assert_has_calls(calls)

    def test_report_watchdog_availability_nonmac(self):
        if False:
            i = 10
            return i + 15
        with patch('streamlit.watcher.path_watcher.watchdog_available', new=False), patch('streamlit.env_util.IS_DARWIN', new=False), patch('click.secho') as mock_echo:
            streamlit.watcher.path_watcher.report_watchdog_availability()
        msg = ''
        calls = [call('  %s' % 'For better performance, install the Watchdog module:', fg='blue', bold=True), call('%s\n  $ pip install watchdog\n            ' % msg)]
        mock_echo.assert_has_calls(calls)

    @patch('streamlit.watcher.path_watcher.PollingPathWatcher')
    @patch('streamlit.watcher.path_watcher.EventBasedPathWatcher')
    def test_watch_file(self, mock_event_watcher, mock_polling_watcher):
        if False:
            return 10
        'Test all possible outcomes of both `get_default_path_watcher_class` and\n        `watch_file`, based on config.fileWatcherType and whether\n        `watchdog_available` is true.\n        '
        subtest_params = [(None, False, NoOpPathWatcher), (None, True, NoOpPathWatcher), ('poll', False, mock_polling_watcher), ('poll', True, mock_polling_watcher), ('watchdog', False, NoOpPathWatcher), ('watchdog', True, mock_event_watcher), ('auto', False, mock_polling_watcher), ('auto', True, mock_event_watcher)]
        for (watcher_config, watchdog_available, path_watcher_class) in subtest_params:
            test_name = f'config.fileWatcherType={watcher_config}, watcher_available={watchdog_available}'
            with self.subTest(test_name):
                with patch_config_options({'server.fileWatcherType': watcher_config}), patch('streamlit.watcher.path_watcher.watchdog_available', watchdog_available):
                    self.assertEqual(path_watcher_class, get_default_path_watcher_class())
                    on_file_changed = Mock()
                    watching_file = watch_file('some/file/path', on_file_changed)
                    if path_watcher_class is not NoOpPathWatcher:
                        path_watcher_class.assert_called_with('some/file/path', on_file_changed, glob_pattern=None, allow_nonexistent=False)
                        self.assertTrue(watching_file)
                    else:
                        self.assertFalse(watching_file)

    @patch('streamlit.watcher.path_watcher.watchdog_available', Mock(return_value=True))
    @patch('streamlit.watcher.path_watcher.EventBasedPathWatcher')
    def test_watch_dir_kwarg_plumbing(self, mock_event_watcher):
        if False:
            return 10
        on_file_changed = Mock()
        watching_dir = watch_dir('some/dir/path', on_file_changed, watcher_type='watchdog', glob_pattern='*.py', allow_nonexistent=True)
        self.assertTrue(watching_dir)
        mock_event_watcher.assert_called_with('some/dir/path', on_file_changed, glob_pattern='*.py', allow_nonexistent=True)