import gc
import io
import logging
import os
import sys
import time
from unittest import TestCase
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class BaseTmpl(TestCase):

    def setUp(self):
        if False:
            return 10
        logging.info('=' * 60)
        logging.info(f'{self.id()} start')
        self.stdout = io.StringIO()
        (self.stdout_orig, sys.stdout) = (sys.stdout, self.stdout)

    def tearDown(self):
        if False:
            while True:
                i = 10
        sys.stdout = self.stdout_orig
        logging.info(f'{self.id()} finish')
        gc.collect()

    def dbgPrint(self, *args, **kwargs):
        if False:
            return 10
        print(*args, file=self.stdout_orig, **kwargs)

    def assertEventNumber(self, data, expected_entries):
        if False:
            while True:
                i = 10
        entries = [entry for entry in data['traceEvents'] if entry['ph'] != 'M']
        entries_count = len(entries)
        self.assertEqual(entries_count, expected_entries, f'Event number incorrect, {entries_count}(expected {expected_entries}) - {entries}')

    def assertFileExists(self, path, timeout=None, msg=None):
        if False:
            for i in range(10):
                print('nop')
        err_msg = f'file {path} does not exist!'
        if msg is not None:
            err_msg = f'file {path} does not exist! {msg}'
        if timeout is None:
            if not os.path.exists(path):
                raise AssertionError(err_msg)
        else:
            start = time.time()
            while True:
                if os.path.exists(path):
                    return
                elif time.time() - start > timeout:
                    raise AssertionError(err_msg)
                else:
                    time.sleep(0.5)

    def assertFileNotExist(self, path):
        if False:
            while True:
                i = 10
        if os.path.exists(path):
            raise AssertionError(f'file {path} does exist!')

    def assertTrueTimeout(self, func, timeout):
        if False:
            return 10
        start = time.time()
        while True:
            try:
                func()
                break
            except AssertionError as e:
                if time.time() - start > timeout:
                    raise e