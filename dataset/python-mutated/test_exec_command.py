import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO

class redirect_stdout:
    """Context manager to redirect stdout for exec_command test."""

    def __init__(self, stdout=None):
        if False:
            while True:
                i = 10
        self._stdout = stdout or sys.stdout

    def __enter__(self):
        if False:
            print('Hello World!')
        self.old_stdout = sys.stdout
        sys.stdout = self._stdout

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        self._stdout.flush()
        sys.stdout = self.old_stdout
        self._stdout.close()

class redirect_stderr:
    """Context manager to redirect stderr for exec_command test."""

    def __init__(self, stderr=None):
        if False:
            i = 10
            return i + 15
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        if False:
            print('Hello World!')
        self.old_stderr = sys.stderr
        sys.stderr = self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self._stderr.flush()
        sys.stderr = self.old_stderr
        self._stderr.close()

class emulate_nonposix:
    """Context manager to emulate os.name != 'posix' """

    def __init__(self, osname='non-posix'):
        if False:
            i = 10
            return i + 15
        self._new_name = osname

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self._old_name = os.name
        os.name = self._new_name

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        os.name = self._old_name

def test_exec_command_stdout():
    if False:
        for i in range(10):
            print('nop')
    with redirect_stdout(StringIO()):
        with redirect_stderr(TemporaryFile()):
            with assert_warns(DeprecationWarning):
                exec_command.exec_command("cd '.'")
    if os.name == 'posix':
        with emulate_nonposix():
            with redirect_stdout(StringIO()):
                with redirect_stderr(TemporaryFile()):
                    with assert_warns(DeprecationWarning):
                        exec_command.exec_command("cd '.'")

def test_exec_command_stderr():
    if False:
        for i in range(10):
            print('nop')
    with redirect_stdout(TemporaryFile(mode='w+')):
        with redirect_stderr(StringIO()):
            with assert_warns(DeprecationWarning):
                exec_command.exec_command("cd '.'")
    if os.name == 'posix':
        with emulate_nonposix():
            with redirect_stdout(TemporaryFile()):
                with redirect_stderr(StringIO()):
                    with assert_warns(DeprecationWarning):
                        exec_command.exec_command("cd '.'")

@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
class TestExecCommand:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.pyexe = get_pythonexe()

    def check_nt(self, **kws):
        if False:
            print('Hello World!')
        (s, o) = exec_command.exec_command('cmd /C echo path=%path%')
        assert_(s == 0)
        assert_(o != '')
        (s, o) = exec_command.exec_command('"%s" -c "import sys;sys.stderr.write(sys.platform)"' % self.pyexe)
        assert_(s == 0)
        assert_(o == 'win32')

    def check_posix(self, **kws):
        if False:
            while True:
                i = 10
        (s, o) = exec_command.exec_command('echo Hello', **kws)
        assert_(s == 0)
        assert_(o == 'Hello')
        (s, o) = exec_command.exec_command('echo $AAA', **kws)
        assert_(s == 0)
        assert_(o == '')
        (s, o) = exec_command.exec_command('echo "$AAA"', AAA='Tere', **kws)
        assert_(s == 0)
        assert_(o == 'Tere')
        (s, o) = exec_command.exec_command('echo "$AAA"', **kws)
        assert_(s == 0)
        assert_(o == '')
        if 'BBB' not in os.environ:
            os.environ['BBB'] = 'Hi'
            (s, o) = exec_command.exec_command('echo "$BBB"', **kws)
            assert_(s == 0)
            assert_(o == 'Hi')
            (s, o) = exec_command.exec_command('echo "$BBB"', BBB='Hey', **kws)
            assert_(s == 0)
            assert_(o == 'Hey')
            (s, o) = exec_command.exec_command('echo "$BBB"', **kws)
            assert_(s == 0)
            assert_(o == 'Hi')
            del os.environ['BBB']
            (s, o) = exec_command.exec_command('echo "$BBB"', **kws)
            assert_(s == 0)
            assert_(o == '')
        (s, o) = exec_command.exec_command('this_is_not_a_command', **kws)
        assert_(s != 0)
        assert_(o != '')
        (s, o) = exec_command.exec_command('echo path=$PATH', **kws)
        assert_(s == 0)
        assert_(o != '')
        (s, o) = exec_command.exec_command('"%s" -c "import sys,os;sys.stderr.write(os.name)"' % self.pyexe, **kws)
        assert_(s == 0)
        assert_(o == 'posix')

    def check_basic(self, *kws):
        if False:
            i = 10
            return i + 15
        (s, o) = exec_command.exec_command('"%s" -c "raise \'Ignore me.\'"' % self.pyexe, **kws)
        assert_(s != 0)
        assert_(o != '')
        (s, o) = exec_command.exec_command('"%s" -c "import sys;sys.stderr.write(\'0\');sys.stderr.write(\'1\');sys.stderr.write(\'2\')"' % self.pyexe, **kws)
        assert_(s == 0)
        assert_(o == '012')
        (s, o) = exec_command.exec_command('"%s" -c "import sys;sys.exit(15)"' % self.pyexe, **kws)
        assert_(s == 15)
        assert_(o == '')
        (s, o) = exec_command.exec_command('"%s" -c "print(\'Heipa\'")' % self.pyexe, **kws)
        assert_(s == 0)
        assert_(o == 'Heipa')

    def check_execute_in(self, **kws):
        if False:
            i = 10
            return i + 15
        with tempdir() as tmpdir:
            fn = 'file'
            tmpfile = os.path.join(tmpdir, fn)
            with open(tmpfile, 'w') as f:
                f.write('Hello')
            (s, o) = exec_command.exec_command('"%s" -c "f = open(\'%s\', \'r\'); f.close()"' % (self.pyexe, fn), **kws)
            assert_(s != 0)
            assert_(o != '')
            (s, o) = exec_command.exec_command('"%s" -c "f = open(\'%s\', \'r\'); print(f.read()); f.close()"' % (self.pyexe, fn), execute_in=tmpdir, **kws)
            assert_(s == 0)
            assert_(o == 'Hello')

    def test_basic(self):
        if False:
            return 10
        with redirect_stdout(StringIO()):
            with redirect_stderr(StringIO()):
                with assert_warns(DeprecationWarning):
                    if os.name == 'posix':
                        self.check_posix(use_tee=0)
                        self.check_posix(use_tee=1)
                    elif os.name == 'nt':
                        self.check_nt(use_tee=0)
                        self.check_nt(use_tee=1)
                    self.check_execute_in(use_tee=0)
                    self.check_execute_in(use_tee=1)