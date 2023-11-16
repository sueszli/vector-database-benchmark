import filecmp
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler
from torch.distributed.elastic.multiprocessing.errors.handlers import get_error_handler

def raise_exception_fn():
    if False:
        return 10
    raise RuntimeError('foobar')

class GetErrorHandlerTest(unittest.TestCase):

    def test_get_error_handler(self):
        if False:
            return 10
        self.assertTrue(isinstance(get_error_handler(), ErrorHandler))

class ErrorHandlerTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self.test_error_file = os.path.join(self.test_dir, 'error.json')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.test_dir)

    @patch('faulthandler.enable')
    def test_initialize(self, fh_enable_mock):
        if False:
            for i in range(10):
                print('nop')
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    @patch('faulthandler.enable', side_effect=RuntimeError)
    def test_initialize_error(self, fh_enable_mock):
        if False:
            print('Hello World!')
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    def test_record_exception(self):
        if False:
            return 10
        with patch.dict(os.environ, {'TORCHELASTIC_ERROR_FILE': self.test_error_file}):
            eh = ErrorHandler()
            eh.initialize()
            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)
            with open(self.test_error_file) as fp:
                err = json.load(fp)
            self.assertIsNotNone(err['message']['message'])
            self.assertIsNotNone(err['message']['extraInfo']['py_callstack'])
            self.assertIsNotNone(err['message']['extraInfo']['timestamp'])

    def test_record_exception_no_error_file(self):
        if False:
            return 10
        with patch.dict(os.environ, {}):
            eh = ErrorHandler()
            eh.initialize()
            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

    def test_dump_error_file(self):
        if False:
            for i in range(10):
                print('nop')
        src_error_file = os.path.join(self.test_dir, 'src_error.json')
        eh = ErrorHandler()
        with patch.dict(os.environ, {'TORCHELASTIC_ERROR_FILE': src_error_file}):
            eh.record_exception(RuntimeError('foobar'))
        with patch.dict(os.environ, {'TORCHELASTIC_ERROR_FILE': self.test_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, self.test_error_file))
        with patch.dict(os.environ, {}):
            eh.dump_error_file(src_error_file)

    def test_dump_error_file_overwrite_existing(self):
        if False:
            for i in range(10):
                print('nop')
        dst_error_file = os.path.join(self.test_dir, 'dst_error.json')
        src_error_file = os.path.join(self.test_dir, 'src_error.json')
        eh = ErrorHandler()
        with patch.dict(os.environ, {'TORCHELASTIC_ERROR_FILE': dst_error_file}):
            eh.record_exception(RuntimeError('foo'))
        with patch.dict(os.environ, {'TORCHELASTIC_ERROR_FILE': src_error_file}):
            eh.record_exception(RuntimeError('bar'))
        with patch.dict(os.environ, {'TORCHELASTIC_ERROR_FILE': dst_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, dst_error_file))