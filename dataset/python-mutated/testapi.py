from __future__ import annotations
import sys
from io import StringIO
import mypy.api
from mypy.test.helpers import Suite

class APISuite(Suite):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        sys.stdout = self.stdout = StringIO()
        sys.stderr = self.stderr = StringIO()

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr
        assert self.stdout.getvalue() == ''
        assert self.stderr.getvalue() == ''

    def test_capture_bad_opt(self) -> None:
        if False:
            i = 10
            return i + 15
        'stderr should be captured when a bad option is passed.'
        (_, stderr, _) = mypy.api.run(['--some-bad-option'])
        assert isinstance(stderr, str)
        assert stderr != ''

    def test_capture_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'stderr should be captured when a bad option is passed.'
        (_, stderr, _) = mypy.api.run([])
        assert isinstance(stderr, str)
        assert stderr != ''

    def test_capture_help(self) -> None:
        if False:
            return 10
        'stdout should be captured when --help is passed.'
        (stdout, _, _) = mypy.api.run(['--help'])
        assert isinstance(stdout, str)
        assert stdout != ''

    def test_capture_version(self) -> None:
        if False:
            while True:
                i = 10
        'stdout should be captured when --version is passed.'
        (stdout, _, _) = mypy.api.run(['--version'])
        assert isinstance(stdout, str)
        assert stdout != ''