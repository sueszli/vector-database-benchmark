"""Utilities for Python Fire's tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import os
import re
import sys
import unittest
from fire import core
from fire import trace
import mock
import six

class BaseTestCase(unittest.TestCase):
    """Shared test case for Python Fire tests."""

    @contextlib.contextmanager
    def assertOutputMatches(self, stdout='.*', stderr='.*', capture=True):
        if False:
            print('Hello World!')
        'Asserts that the context generates stdout and stderr matching regexps.\n\n    Note: If wrapped code raises an exception, stdout and stderr will not be\n      checked.\n\n    Args:\n      stdout: (str) regexp to match against stdout (None will check no stdout)\n      stderr: (str) regexp to match against stderr (None will check no stderr)\n      capture: (bool, default True) do not bubble up stdout or stderr\n\n    Yields:\n      Yields to the wrapped context.\n    '
        stdout_fp = six.StringIO()
        stderr_fp = six.StringIO()
        try:
            with mock.patch.object(sys, 'stdout', stdout_fp):
                with mock.patch.object(sys, 'stderr', stderr_fp):
                    yield
        finally:
            if not capture:
                sys.stdout.write(stdout_fp.getvalue())
                sys.stderr.write(stderr_fp.getvalue())
        for (name, regexp, fp) in [('stdout', stdout, stdout_fp), ('stderr', stderr, stderr_fp)]:
            value = fp.getvalue()
            if regexp is None:
                if value:
                    raise AssertionError('%s: Expected no output. Got: %r' % (name, value))
            elif not re.search(regexp, value, re.DOTALL | re.MULTILINE):
                raise AssertionError('%s: Expected %r to match %r' % (name, value, regexp))

    def assertRaisesRegex(self, *args, **kwargs):
        if False:
            return 10
        if sys.version_info.major == 2:
            return super(BaseTestCase, self).assertRaisesRegexp(*args, **kwargs)
        else:
            return super(BaseTestCase, self).assertRaisesRegex(*args, **kwargs)

    @contextlib.contextmanager
    def assertRaisesFireExit(self, code, regexp='.*'):
        if False:
            while True:
                i = 10
        "Asserts that a FireExit error is raised in the context.\n\n    Allows tests to check that Fire's wrapper around SystemExit is raised\n    and that a regexp is matched in the output.\n\n    Args:\n      code: The status code that the FireExit should contain.\n      regexp: stdout must match this regex.\n\n    Yields:\n      Yields to the wrapped context.\n    "
        with self.assertOutputMatches(stderr=regexp):
            with self.assertRaises(core.FireExit):
                try:
                    yield
                except core.FireExit as exc:
                    if exc.code != code:
                        raise AssertionError('Incorrect exit code: %r != %r' % (exc.code, code))
                    self.assertIsInstance(exc.trace, trace.FireTrace)
                    raise

@contextlib.contextmanager
def ChangeDirectory(directory):
    if False:
        for i in range(10):
            print('nop')
    'Context manager to mock a directory change and revert on exit.'
    cwdir = os.getcwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(cwdir)
main = unittest.main
skip = unittest.skip
skipIf = unittest.skipIf