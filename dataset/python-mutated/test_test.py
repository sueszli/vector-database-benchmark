import subprocess
from textwrap import dedent
import pytest

def test_test_pass(tmpdir, capsys):
    if False:
        for i in range(10):
            print('nop')
    tmpdir.join('__init__.py')
    testfile = tmpdir.join('test_test_pass.py')
    testfile.write(dedent('\n        def test_foo():\n            assert True\n    '))
    proc = subprocess.Popen(['nameko', 'test', testfile.strpath], stdout=subprocess.PIPE)
    proc.wait()
    assert proc.returncode == 0

def test_test_fail(tmpdir, capsys):
    if False:
        for i in range(10):
            print('nop')
    tmpdir.join('__init__.py')
    testfile = tmpdir.join('test_test_fail.py')
    testfile.write(dedent('\n        def test_foo():\n            assert False\n    '))
    proc = subprocess.Popen(['nameko', 'test', testfile.strpath], stdout=subprocess.PIPE)
    proc.wait()
    assert proc.returncode == 1

def test_suppress_warning(tmpdir, capsys):
    if False:
        print('Hello World!')
    if tuple(map(int, pytest.__version__.split('.'))) < (6, 1):
        pytest.skip('-W flag ignored on older pytests')
    tmpdir.join('__init__.py')
    testfile = tmpdir.join('test_test_pass.py')
    testfile.write(dedent('\n        def test_foo():\n            assert True\n    '))
    proc = subprocess.Popen(['nameko', 'test', testfile.strpath], stdout=subprocess.PIPE)
    proc.wait()
    out = ''.join(map(bytes.decode, proc.stdout.readlines()))
    assert 'Module already imported so cannot be rewritten: nameko' not in out