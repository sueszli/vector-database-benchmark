import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging

@contextlib.contextmanager
def preserve_log_state():
    if False:
        for i in range(10):
            print('nop')
    prev_state = torch._logging._internal._get_log_state()
    torch._logging._internal._set_log_state(torch._logging._internal.LogState())
    try:
        yield
    finally:
        torch._logging._internal._set_log_state(prev_state)
        torch._logging._internal._init_logs()

def log_settings(settings):
    if False:
        i = 10
        return i + 15
    exit_stack = contextlib.ExitStack()
    settings_patch = unittest.mock.patch.dict(os.environ, {'TORCH_LOGS': settings})
    exit_stack.enter_context(preserve_log_state())
    exit_stack.enter_context(settings_patch)
    torch._logging._internal._init_logs()
    return exit_stack

def log_api(**kwargs):
    if False:
        while True:
            i = 10
    exit_stack = contextlib.ExitStack()
    exit_stack.enter_context(preserve_log_state())
    torch._logging.set_logs(**kwargs)
    return exit_stack

def kwargs_to_settings(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    INT_TO_VERBOSITY = {10: '+', 20: '', 40: '-'}
    settings = []

    def append_setting(name, level):
        if False:
            return 10
        if isinstance(name, str) and isinstance(level, int) and (level in INT_TO_VERBOSITY):
            settings.append(INT_TO_VERBOSITY[level] + name)
            return
        else:
            raise ValueError('Invalid value for setting')
    for (name, val) in kwargs.items():
        if isinstance(val, bool):
            settings.append(name)
        elif isinstance(val, int):
            append_setting(name, val)
        elif isinstance(val, dict) and name == 'modules':
            for (module_qname, level) in val.items():
                append_setting(module_qname, level)
        else:
            raise ValueError('Invalid value for setting')
    return ','.join(settings)

def make_logging_test(**kwargs):
    if False:
        while True:
            i = 10

    def wrapper(fn):
        if False:
            i = 10
            return i + 15

        def test_fn(self):
            if False:
                while True:
                    i = 10
            torch._dynamo.reset()
            records = []
            if len(kwargs) == 0:
                with self._handler_watcher(records):
                    fn(self, records)
            else:
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    fn(self, records)
            torch._dynamo.reset()
            records.clear()
            with log_api(**kwargs), self._handler_watcher(records):
                fn(self, records)
        return test_fn
    return wrapper

def make_settings_test(settings):
    if False:
        while True:
            i = 10

    def wrapper(fn):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(self):
            if False:
                while True:
                    i = 10
            torch._dynamo.reset()
            records = []
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)
        return test_fn
    return wrapper

class LoggingTestCase(torch._dynamo.test_case.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls._exit_stack.enter_context(unittest.mock.patch.dict(os.environ, {'___LOG_TESTING': ''}))
        cls._exit_stack.enter_context(unittest.mock.patch('torch._dynamo.config.suppress_errors', True))
        cls._exit_stack.enter_context(unittest.mock.patch('torch._dynamo.config.verbose', False))

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls._exit_stack.close()
        torch._logging._internal.log_state.clear()
        torch._logging._init_logs()

    def getRecord(self, records, m):
        if False:
            while True:
                i = 10
        record = None
        for r in records:
            if m in r.getMessage():
                self.assertIsNone(record, msg=LazyString(lambda : f'multiple matching records: {record} and {r} among {records}'))
                record = r
        if record is None:
            self.fail(f'did not find record with {m} among {records}')
        return record

    def _handler_watcher(self, record_list):
        if False:
            i = 10
            return i + 15
        exit_stack = contextlib.ExitStack()

        def emit_post_hook(record):
            if False:
                i = 10
                return i + 15
            nonlocal record_list
            record_list.append(record)
        for log_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(log_qname)
            num_handlers = len(logger.handlers)
            self.assertLessEqual(num_handlers, 2, 'All pt2 loggers should only have at most two handlers (debug artifacts and messages above debug level).')
            self.assertGreater(num_handlers, 0, 'All pt2 loggers should have more than zero handlers')
            for handler in logger.handlers:
                old_emit = handler.emit

                def new_emit(record):
                    if False:
                        while True:
                            i = 10
                    old_emit(record)
                    emit_post_hook(record)
                exit_stack.enter_context(unittest.mock.patch.object(handler, 'emit', new_emit))
        return exit_stack