import os
import subprocess
import tempfile
import mock
from helpers import unittest
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential
import luigi
import luigi.lock
import luigi.notifications
luigi.notifications.DEBUG = True

class TestCmd(unittest.TestCase):

    def test_getpcmd(self):
        if False:
            return 10

        def _is_empty(cmd):
            if False:
                print('Hello World!')
            return cmd == ''

        @retry(retry=retry_if_result(_is_empty), wait=wait_exponential(multiplier=0.2, min=0.1, max=3), stop=stop_after_attempt(3))
        def _getpcmd(pid):
            if False:
                for i in range(10):
                    print('nop')
            return luigi.lock.getpcmd(pid)
        if os.name == 'nt':
            command = ['ping', '1.1.1.1', '-w', '1000']
        else:
            command = ['sleep', '1']
        external_process = subprocess.Popen(command)
        result = _getpcmd(external_process.pid)
        self.assertTrue(result.strip() in ['sleep 1', '[sleep]', 'ping 1.1.1.1 -w 1000'])
        external_process.kill()

class LockTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.pid_dir = tempfile.mkdtemp()
        (self.pid, self.cmd, self.pid_file) = luigi.lock.get_info(self.pid_dir)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)
        os.rmdir(self.pid_dir)

    def test_get_info(self):
        if False:
            print('Hello World!')

        def _is_empty(result):
            if False:
                i = 10
                return i + 15
            return result[1] == ''

        @retry(retry=retry_if_result(_is_empty), wait=wait_exponential(multiplier=0.2, min=0.1, max=3), stop=stop_after_attempt(3))
        def _get_info(pid_dir, pid):
            if False:
                for i in range(10):
                    print('nop')
            return luigi.lock.get_info(pid_dir, pid)
        try:
            p = subprocess.Popen(['yes', u'à我ф'], stdout=subprocess.PIPE)
            (pid, cmd, pid_file) = _get_info(self.pid_dir, p.pid)
        finally:
            p.kill()
        self.assertEqual(cmd, u'yes à我ф')

    def test_acquiring_free_lock(self):
        if False:
            return 10
        acquired = luigi.lock.acquire_for(self.pid_dir)
        self.assertTrue(acquired)

    def test_acquiring_taken_lock(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.pid_file, 'w') as f:
            f.write('%d\n' % (self.pid,))
        acquired = luigi.lock.acquire_for(self.pid_dir)
        self.assertFalse(acquired)

    def test_acquiring_partially_taken_lock(self):
        if False:
            i = 10
            return i + 15
        with open(self.pid_file, 'w') as f:
            f.write('%d\n' % (self.pid,))
        acquired = luigi.lock.acquire_for(self.pid_dir, 2)
        self.assertTrue(acquired)
        s = os.stat(self.pid_file)
        self.assertEqual(s.st_mode & 511, 511)

    def test_acquiring_lock_from_missing_process(self):
        if False:
            for i in range(10):
                print('nop')
        fake_pid = 99999
        with open(self.pid_file, 'w') as f:
            f.write('%d\n' % (fake_pid,))
        acquired = luigi.lock.acquire_for(self.pid_dir)
        self.assertTrue(acquired)
        s = os.stat(self.pid_file)
        self.assertEqual(s.st_mode & 511, 511)

    @mock.patch('os.kill')
    def test_take_lock_with_kill(self, kill_fn):
        if False:
            i = 10
            return i + 15
        with open(self.pid_file, 'w') as f:
            f.write('%d\n' % (self.pid,))
        kill_signal = 77777
        acquired = luigi.lock.acquire_for(self.pid_dir, kill_signal=kill_signal)
        self.assertTrue(acquired)
        kill_fn.assert_called_once_with(self.pid, kill_signal)

    @mock.patch('os.kill')
    @mock.patch('luigi.lock.getpcmd')
    def test_take_lock_has_only_one_extra_life(self, getpcmd, kill_fn):
        if False:
            print('Hello World!')

        def side_effect(pid):
            if False:
                i = 10
                return i + 15
            if pid in [self.pid, self.pid + 1, self.pid + 2]:
                return self.cmd
            else:
                return 'echo something_else'
        getpcmd.side_effect = side_effect
        with open(self.pid_file, 'w') as f:
            f.write('{}\n{}\n'.format(self.pid + 1, self.pid + 2))
        kill_signal = 77777
        acquired = luigi.lock.acquire_for(self.pid_dir, kill_signal=kill_signal)
        self.assertFalse(acquired)
        kill_fn.assert_any_call(self.pid + 1, kill_signal)
        kill_fn.assert_any_call(self.pid + 2, kill_signal)

    @mock.patch('luigi.lock.getpcmd')
    def test_cleans_old_pid_entries(self, getpcmd):
        if False:
            i = 10
            return i + 15
        assert self.pid > 10
        SAME_ENTRIES = {1, 2, 3, 4, 5, self.pid}
        ALL_ENTRIES = SAME_ENTRIES | {6, 7, 8, 9, 10}

        def side_effect(pid):
            if False:
                for i in range(10):
                    print('nop')
            if pid in SAME_ENTRIES:
                return self.cmd
            elif pid == 8:
                return None
            else:
                return 'echo something_else'
        getpcmd.side_effect = side_effect
        with open(self.pid_file, 'w') as f:
            f.writelines(('{}\n'.format(pid) for pid in ALL_ENTRIES))
        acquired = luigi.lock.acquire_for(self.pid_dir, num_available=100)
        self.assertTrue(acquired)
        with open(self.pid_file, 'r') as f:
            self.assertEqual({int(pid_str.strip()) for pid_str in f}, SAME_ENTRIES)