import os
import subprocess
import tempfile
import time
import pytest
import platformdirs
from . import exec_cmd
from ...platformflags import is_win32
from ...helpers import get_runtime_dir

def have_a_short_runtime_dir(mp):
    if False:
        for i in range(10):
            print('nop')
    mp.setenv('BORG_RUNTIME_DIR', os.path.join(platformdirs.user_runtime_dir(), 'pytest'))

@pytest.fixture
def serve_socket(monkeypatch):
    if False:
        print('Hello World!')
    have_a_short_runtime_dir(monkeypatch)
    socket_file = tempfile.mktemp(suffix='.sock', prefix='borg-', dir=get_runtime_dir())
    with subprocess.Popen(['borg', 'serve', f'--socket={socket_file}']) as p:
        while not os.path.exists(socket_file):
            time.sleep(0.01)
        yield socket_file
        p.terminate()

@pytest.mark.skipif(is_win32, reason='hangs on win32')
def test_with_socket(serve_socket, tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    have_a_short_runtime_dir(monkeypatch)
    repo_path = str(tmpdir.join('repo'))
    (ret, output) = exec_cmd(f'--socket={serve_socket}', f'--repo=socket://{repo_path}', 'rcreate', '--encryption=none')
    assert ret == 0
    (ret, output) = exec_cmd(f'--socket={serve_socket}', f'--repo=socket://{repo_path}', 'rinfo')
    assert ret == 0
    assert 'Repository ID: ' in output
    monkeypatch.setenv('BORG_DELETE_I_KNOW_WHAT_I_AM_DOING', 'YES')
    (ret, output) = exec_cmd(f'--socket={serve_socket}', f'--repo=socket://{repo_path}', 'rdelete')
    assert ret == 0

@pytest.mark.skipif(is_win32, reason='hangs on win32')
def test_socket_permissions(serve_socket):
    if False:
        while True:
            i = 10
    st = os.stat(serve_socket)
    assert st.st_mode & 511 == 504