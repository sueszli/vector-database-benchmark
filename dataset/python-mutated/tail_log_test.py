import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from concurrent.futures import wait
from concurrent.futures._base import ALL_COMPLETED
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict, Set
from unittest import mock
from torch.distributed.elastic.multiprocessing.tail_log import TailLog

def write(max: int, sleep: float, file: str):
    if False:
        while True:
            i = 10
    with open(file, 'w') as fp:
        for i in range(max):
            print(i, file=fp, flush=True)
            time.sleep(sleep)

class TailLogTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_dir = tempfile.mkdtemp(prefix=f'{self.__class__.__name__}_')
        self.threadpool = ThreadPoolExecutor()

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.test_dir)

    def test_tail(self):
        if False:
            i = 10
            return i + 15
        '\n        writer() writes 0 - max (on number on each line) to a log file.\n        Run nprocs such writers and tail the log files into an IOString\n        and validate that all lines are accounted for.\n        '
        nprocs = 32
        max = 1000
        interval_sec = 0.0001
        log_files = {local_rank: os.path.join(self.test_dir, f'{local_rank}_stdout.log') for local_rank in range(nprocs)}
        dst = io.StringIO()
        tail = TailLog(name='writer', log_files=log_files, dst=dst, interval_sec=interval_sec).start()
        time.sleep(interval_sec * 10)
        futs = []
        for (local_rank, file) in log_files.items():
            f = self.threadpool.submit(write, max=max, sleep=interval_sec * local_rank, file=file)
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)
        actual: Dict[int, Set[int]] = {}
        for line in dst.readlines():
            (header, num) = line.split(':')
            nums = actual.setdefault(header, set())
            nums.add(int(num))
        self.assertEqual(nprocs, len(actual))
        self.assertEqual({f'[writer{i}]': set(range(max)) for i in range(nprocs)}, actual)
        self.assertTrue(tail.stopped())

    def test_tail_with_custom_prefix(self):
        if False:
            while True:
                i = 10
        '\n        writer() writes 0 - max (on number on each line) to a log file.\n        Run nprocs such writers and tail the log files into an IOString\n        and validate that all lines are accounted for.\n        '
        nprocs = 3
        max = 10
        interval_sec = 0.0001
        log_files = {local_rank: os.path.join(self.test_dir, f'{local_rank}_stdout.log') for local_rank in range(nprocs)}
        dst = io.StringIO()
        log_line_prefixes = {n: f'[worker{n}][{n}]:' for n in range(nprocs)}
        tail = TailLog('writer', log_files, dst, interval_sec=interval_sec, log_line_prefixes=log_line_prefixes).start()
        time.sleep(interval_sec * 10)
        futs = []
        for (local_rank, file) in log_files.items():
            f = self.threadpool.submit(write, max=max, sleep=interval_sec * local_rank, file=file)
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)
        headers: Set[str] = set()
        for line in dst.readlines():
            (header, _) = line.split(':')
            headers.add(header)
        self.assertEqual(nprocs, len(headers))
        for i in range(nprocs):
            self.assertIn(f'[worker{i}][{i}]', headers)
        self.assertTrue(tail.stopped())

    def test_tail_no_files(self):
        if False:
            return 10
        '\n        Ensures that the log tail can gracefully handle no log files\n        in which case it does nothing.\n        '
        tail = TailLog('writer', log_files={}, dst=sys.stdout).start()
        self.assertFalse(tail.stopped())
        tail.stop()
        self.assertTrue(tail.stopped())

    def test_tail_logfile_never_generates(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensures that we properly shutdown the threadpool\n        even when the logfile never generates.\n        '
        tail = TailLog('writer', log_files={0: 'foobar.log'}, dst=sys.stdout).start()
        tail.stop()
        self.assertTrue(tail.stopped())
        self.assertTrue(tail._threadpool._shutdown)

    @mock.patch('torch.distributed.elastic.multiprocessing.tail_log.log')
    def test_tail_logfile_error_in_tail_fn(self, mock_logger):
        if False:
            i = 10
            return i + 15
        '\n        Ensures that when there is an error in the tail_fn (the one that runs in the\n        threadpool), it is dealt with and raised properly.\n        '
        tail = TailLog('writer', log_files={0: self.test_dir}, dst=sys.stdout).start()
        tail.stop()
        mock_logger.error.assert_called_once()