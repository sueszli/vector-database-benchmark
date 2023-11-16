from __future__ import annotations
import sys
import json
from contextlib import contextmanager
from io import BytesIO, StringIO
import unittest
from ansible.module_utils.common.text.converters import to_bytes

@contextmanager
def swap_stdin_and_argv(stdin_data='', argv_data=tuple()):
    if False:
        i = 10
        return i + 15
    "\n    context manager that temporarily masks the test runner's values for stdin and argv\n    "
    real_stdin = sys.stdin
    real_argv = sys.argv
    fake_stream = StringIO(stdin_data)
    fake_stream.buffer = BytesIO(to_bytes(stdin_data))
    try:
        sys.stdin = fake_stream
        sys.argv = argv_data
        yield
    finally:
        sys.stdin = real_stdin
        sys.argv = real_argv

class ModuleTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        module_args = {'_ansible_remote_tmp': '/tmp', '_ansible_keep_remote_files': False}
        args = json.dumps(dict(ANSIBLE_MODULE_ARGS=module_args))
        self.stdin_swap = swap_stdin_and_argv(stdin_data=args)
        self.stdin_swap.__enter__()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.stdin_swap.__exit__(None, None, None)