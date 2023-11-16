import os
import subprocess
import sys
from shutil import which
import pytest
GDB_COMMANDS = '\nset confirm off\nset breakpoint pending on\nbreak cuInit\nrun\nexit\n'

@pytest.fixture(scope='module')
def cuda_gdb(request):
    if False:
        i = 10
        return i + 15
    gdb = which('cuda-gdb')
    if gdb is None:
        request.applymarker(pytest.mark.xfail(reason="No cuda-gdb found, can't detect cuInit"))
        return gdb
    else:
        output = subprocess.run([gdb, '--version'], capture_output=True, text=True, cwd='/')
        if output.returncode != 0:
            request.applymarker(pytest.mark.xfail(reason=f"cuda-gdb not working on this platform, can't detect cuInit: {output.stderr}"))
        return gdb

def test_cudf_import_no_cuinit(cuda_gdb):
    if False:
        print('Hello World!')
    env = os.environ.copy()
    env['RAPIDS_NO_INITIALIZE'] = '1'
    output = subprocess.run([cuda_gdb, '-x', '-', '--args', sys.executable, '-c', 'import cudf'], input=GDB_COMMANDS, env=env, capture_output=True, text=True, cwd='/')
    cuInit_called = output.stdout.find('in cuInit ()')
    print('Command output:\n')
    print('*** STDOUT ***')
    print(output.stdout)
    print('*** STDERR ***')
    print(output.stderr)
    assert output.returncode == 0
    assert cuInit_called < 0

def test_cudf_create_series_cuinit(cuda_gdb):
    if False:
        print('Hello World!')
    env = os.environ.copy()
    env['RAPIDS_NO_INITIALIZE'] = '1'
    output = subprocess.run([cuda_gdb, '-x', '-', '--args', sys.executable, '-c', 'import cudf; cudf.Series([1])'], input=GDB_COMMANDS, env=env, capture_output=True, text=True, cwd='/')
    cuInit_called = output.stdout.find('in cuInit ()')
    print('Command output:\n')
    print('*** STDOUT ***')
    print(output.stdout)
    print('*** STDERR ***')
    print(output.stderr)
    assert output.returncode == 0
    assert cuInit_called >= 0