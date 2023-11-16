from unittest import TestCase
import plotly.io as pio
import subprocess
import os
from packaging.version import Version
import requests
import time
import psutil
import pytest
import plotly.graph_objects as go
from plotly.io._orca import find_open_port, which, orca_env

@pytest.fixture()
def setup():
    if False:
        return 10
    os.environ['NODE_OPTIONS'] = '--max-old-space-size=4096'
    os.environ['ELECTRON_RUN_AS_NODE'] = '1'
    pio.orca.reset_status()
    pio.orca.config.restore_defaults()
pytestmark = pytest.mark.usefixtures('setup')

def ping_pongs(server_url):
    if False:
        while True:
            i = 10
    try:
        response = requests.post(server_url + '/ping')
    except requests.exceptions.ConnectionError:
        return False
    return response.status_code == 200 and response.content.decode('utf-8') == 'pong'

def test_validate_orca():
    if False:
        for i in range(10):
            print('nop')
    assert pio.orca.status.state == 'unvalidated'
    pio.orca.validate_executable()
    assert pio.orca.status.state == 'validated'

def test_orca_not_found():
    if False:
        i = 10
        return i + 15
    pio.orca.config.executable = 'bogus'
    with pytest.raises(ValueError) as err:
        pio.orca.validate_executable()
    assert pio.orca.status.state == 'unvalidated'
    assert 'could not be found' in str(err.value)

def test_invalid_executable_found():
    if False:
        for i in range(10):
            print('nop')
    pio.orca.config.executable = 'python'
    with pytest.raises(ValueError) as err:
        pio.orca.validate_executable()
    assert pio.orca.status.state == 'unvalidated'
    assert 'executable that was found at' in str(err.value)

def test_orca_executable_path():
    if False:
        for i in range(10):
            print('nop')
    assert pio.orca.status.executable is None
    if os.name == 'nt':
        expected = subprocess.check_output(['where', 'orca']).decode('utf-8').strip()
    else:
        expected = subprocess.check_output(['which', 'orca']).decode('utf-8').strip()
    pio.orca.validate_executable()
    assert pio.orca.status.executable == expected

def test_orca_version_number():
    if False:
        while True:
            i = 10
    assert pio.orca.status.version is None
    expected_min = Version('1.1.0')
    expected_max = Version('2.0.0')
    pio.orca.validate_executable()
    version = Version(pio.orca.status.version)
    assert expected_min <= version
    assert version < expected_max

def test_ensure_orca_ping_and_proc():
    if False:
        i = 10
        return i + 15
    pio.orca.config.timeout = None
    assert pio.orca.status.port is None
    assert pio.orca.status.pid is None
    pio.orca.ensure_server()
    assert pio.orca.status.port is not None
    assert pio.orca.status.pid is not None
    server_port = pio.orca.status.port
    server_pid = pio.orca.status.pid
    time.sleep(10)
    assert psutil.pid_exists(server_pid)
    server_url = 'http://localhost:%s' % server_port
    assert ping_pongs(server_url)
    pio.orca.shutdown_server()
    assert not psutil.pid_exists(server_pid)
    assert not ping_pongs(server_url)

def test_server_timeout_shutdown():
    if False:
        print('Hello World!')
    pio.orca.config.timeout = 10
    pio.orca.ensure_server()
    server_port = pio.orca.status.port
    server_pid = pio.orca.status.pid
    server_url = 'http://localhost:%s' % server_port
    assert psutil.pid_exists(server_pid)
    for i in range(3):
        time.sleep(8)
        assert ping_pongs(server_url)
        assert psutil.pid_exists(server_pid)
        pio.orca.ensure_server()
    time.sleep(11)
    assert not psutil.pid_exists(server_pid)
    assert not ping_pongs(server_url)

def test_external_server_url():
    if False:
        return 10
    port = find_open_port()
    server_url = 'http://{hostname}:{port}'.format(hostname='localhost', port=port)
    orca_path = which('orca')
    cmd_list = [orca_path] + ['serve', '-p', str(port), '--plotly', pio.orca.config.plotlyjs, '--graph-only']
    DEVNULL = open(os.devnull, 'wb')
    with orca_env():
        proc = subprocess.Popen(cmd_list, stdout=DEVNULL)
    pio.orca.config.port = port
    pio.orca.ensure_server()
    assert pio.orca.status.state == 'running'
    pio.orca.config.server_url = server_url
    assert pio.orca.status.state == 'unvalidated'
    assert pio.orca.config.port is None
    fig = go.Figure()
    img_bytes = pio.to_image(fig, format='svg')
    assert img_bytes.startswith(b'<svg class')
    proc.terminate()