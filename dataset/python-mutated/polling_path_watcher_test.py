import unittest
from unittest import mock
from streamlit.watcher import polling_path_watcher

class PollingPathWatcherTest(unittest.TestCase):
    """Test PollingPathWatcher."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(PollingPathWatcherTest, self).setUp()
        self.util_patch = mock.patch('streamlit.watcher.polling_path_watcher.util')
        self.util_mock = self.util_patch.start()
        self._executor_tasks = []
        self.executor_patch = mock.patch('streamlit.watcher.polling_path_watcher.PollingPathWatcher._executor')
        executor_mock = self.executor_patch.start()
        executor_mock.submit = self._submit_executor_task
        self.sleep_patch = mock.patch('streamlit.watcher.polling_path_watcher.time.sleep')
        self.sleep_patch.start()

    def tearDown(self):
        if False:
            return 10
        super(PollingPathWatcherTest, self).tearDown()
        self.util_patch.stop()
        self.executor_patch.stop()
        self.sleep_patch.stop()

    def _submit_executor_task(self, task):
        if False:
            print('Hello World!')
        'Submit a new task to our mock executor.'
        self._executor_tasks.append(task)

    def _run_executor_tasks(self):
        if False:
            while True:
                i = 10
        'Run all tasks that have been submitted to our mock executor.'
        tasks = self._executor_tasks
        self._executor_tasks = []
        for task in tasks:
            task()

    def test_file_watch_and_callback(self):
        if False:
            print('Hello World!')
        'Test that when a file is modified, the callback is called.'
        callback = mock.Mock()
        self.util_mock.path_modification_time = lambda *args: 101.0
        self.util_mock.calc_md5_with_blocking_retries = lambda _, **kwargs: '1'
        watcher = polling_path_watcher.PollingPathWatcher('/this/is/my/file.py', callback)
        self._run_executor_tasks()
        callback.assert_not_called()
        self.util_mock.path_modification_time = lambda *args: 102.0
        self.util_mock.calc_md5_with_blocking_retries = lambda _, **kwargs: '2'
        self._run_executor_tasks()
        callback.assert_called_once()
        watcher.close()

    def test_callback_not_called_if_same_mtime(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we ignore files with same mtime.'
        callback = mock.Mock()
        self.util_mock.path_modification_time = lambda *args: 101.0
        self.util_mock.calc_md5_with_blocking_retries = lambda _, **kwargs: '1'
        watcher = polling_path_watcher.PollingPathWatcher('/this/is/my/file.py', callback)
        self._run_executor_tasks()
        callback.assert_not_called()
        self.util_mock.calc_md5_with_blocking_retries = lambda _, **kwargs: '2'
        self._run_executor_tasks()
        callback.assert_not_called()
        watcher.close()

    def test_callback_not_called_if_same_md5(self):
        if False:
            print('Hello World!')
        'Test that we ignore files with same md5.'
        callback = mock.Mock()
        self.util_mock.path_modification_time = lambda *args: 101.0
        self.util_mock.calc_md5_with_blocking_retries = lambda _, **kwargs: '1'
        watcher = polling_path_watcher.PollingPathWatcher('/this/is/my/file.py', callback)
        self._run_executor_tasks()
        callback.assert_not_called()
        self.util_mock.path_modification_time = lambda *args: 102.0
        self._run_executor_tasks()
        callback.assert_not_called()
        watcher.close()

    def test_kwargs_plumbed_to_calc_md5(self):
        if False:
            print('Hello World!')
        'Test that we pass the glob_pattern and allow_nonexistent kwargs to\n        calc_md5_with_blocking_retries.\n\n        `PollingPathWatcher`s can be created with optional kwargs allowing\n        the caller to specify what types of files to watch (when watching a\n        directory) and whether to allow watchers on paths with no files/dirs.\n        This test ensures that these optional parameters make it to our hash\n        calculation helpers across different on_changed events.\n        '
        callback = mock.Mock()
        self.util_mock.path_modification_time = lambda *args: 101.0
        self.util_mock.calc_md5_with_blocking_retries = mock.Mock(return_value='1')
        watcher = polling_path_watcher.PollingPathWatcher('/this/is/my/dir', callback, glob_pattern='*.py', allow_nonexistent=True)
        self._run_executor_tasks()
        callback.assert_not_called()
        (_, kwargs) = self.util_mock.calc_md5_with_blocking_retries.call_args
        assert kwargs == {'glob_pattern': '*.py', 'allow_nonexistent': True}
        self.util_mock.path_modification_time = lambda *args: 102.0
        self.util_mock.calc_md5_with_blocking_retries = mock.Mock(return_value='2')
        self._run_executor_tasks()
        callback.assert_called_once()
        (_, kwargs) = self.util_mock.calc_md5_with_blocking_retries.call_args
        assert kwargs == {'glob_pattern': '*.py', 'allow_nonexistent': True}
        watcher.close()

    def test_multiple_watchers_same_file(self):
        if False:
            print('Hello World!')
        'Test that we can have multiple watchers of the same file.'
        filename = '/this/is/my/file.py'
        mod_count = [0.0]

        def modify_mock_file():
            if False:
                print('Hello World!')
            self.util_mock.path_modification_time = lambda *args: mod_count[0]
            self.util_mock.calc_md5_with_blocking_retries = lambda _, **kwargs: '%d' % mod_count[0]
            mod_count[0] += 1.0
        modify_mock_file()
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        watcher1 = polling_path_watcher.PollingPathWatcher(filename, callback1)
        watcher2 = polling_path_watcher.PollingPathWatcher(filename, callback2)
        self._run_executor_tasks()
        callback1.assert_not_called()
        callback2.assert_not_called()
        modify_mock_file()
        self._run_executor_tasks()
        self.assertEqual(callback1.call_count, 1)
        self.assertEqual(callback2.call_count, 1)
        watcher1.close()
        modify_mock_file()
        self._run_executor_tasks()
        self.assertEqual(callback1.call_count, 1)
        self.assertEqual(callback2.call_count, 2)
        watcher2.close()
        modify_mock_file()
        self.assertEqual(callback1.call_count, 1)
        self.assertEqual(callback2.call_count, 2)