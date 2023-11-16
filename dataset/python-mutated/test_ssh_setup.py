"""
Integration tests for salt-ssh py_versions
"""
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import pytest
from pytestshellutils.utils.processes import ProcessResult, terminate_process
from saltfactories.utils import random_string
from tests.support.helpers import Keys
pytest.importorskip('docker')
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.slow_test, pytest.mark.skip_if_binaries_missing('dockerd')]

@pytest.fixture(scope='module')
def ssh_keys(tmp_path_factory):
    if False:
        print('Hello World!')
    '\n    Temporary ssh key fixture\n    '
    with Keys(tmp_path_factory) as keys:
        yield keys

@pytest.fixture(scope='module')
def ssh_password():
    if False:
        while True:
            i = 10
    return random_string('sshpassword')

@pytest.fixture(scope='module')
def ssh_container_name():
    if False:
        i = 10
        return i + 15
    return random_string('ssh-container-')

@pytest.fixture(scope='module')
def ssh_sub_container_name():
    if False:
        print('Hello World!')
    return random_string('ssh-sub-container-')

@pytest.fixture(scope='module')
def ssh_container(salt_factories, ssh_container_name, ssh_password):
    if False:
        for i in range(10):
            print('nop')
    '\n    Temporary docker container with python 3.6 and ssh enabled\n    '
    container = salt_factories.get_container(ssh_container_name, 'dwoz1/cicd:ssh', container_run_kwargs={'ports': {'22/tcp': None}, 'environment': {'SSH_USER': 'centos', 'SSH_USER_PASSWORD': ssh_password, 'SSH_PASSWORD_AUTHENTICATION': 'true'}, 'cap_add': 'IPC_LOCK'}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def ssh_sub_container(salt_factories, ssh_sub_container_name, ssh_password):
    if False:
        print('Hello World!')
    '\n    Temporary docker container with python 3.6 and ssh enabled\n    '
    container = salt_factories.get_container(ssh_sub_container_name, 'dwoz1/cicd:ssh', container_run_kwargs={'ports': {'22/tcp': None}, 'environment': {'SSH_USER': 'centos', 'SSH_USER_PASSWORD': ssh_password, 'SSH_PASSWORD_AUTHENTICATION': 'true'}, 'cap_add': 'IPC_LOCK'}, pull_before_start=True, skip_on_pull_failure=True, skip_if_docker_client_not_connectable=True)
    with container.started() as factory:
        yield factory

@pytest.fixture(scope='module')
def ssh_port(ssh_container):
    if False:
        while True:
            i = 10
    return ssh_container.get_host_port_binding(22, protocol='tcp')

@pytest.fixture(scope='module')
def ssh_sub_port(ssh_sub_container):
    if False:
        while True:
            i = 10
    return ssh_sub_container.get_host_port_binding(22, protocol='tcp')

@pytest.fixture(scope='module')
def salt_ssh_roster_file(ssh_container_name, ssh_sub_container_name, ssh_port, ssh_sub_port, salt_master):
    if False:
        for i in range(10):
            print('nop')
    '\n    Temporary roster for ssh docker container\n    '
    roster = '\n    {}:\n      host: localhost\n      user: centos\n      port: {}\n    {}:\n      host: localhost\n      user: centos\n      port: {}\n    '.format(ssh_container_name, ssh_port, ssh_sub_container_name, ssh_sub_port)
    with pytest.helpers.temp_file('setup_roster', roster, salt_master.config_dir) as roster_file:
        yield roster_file

@pytest.fixture(scope='module')
def salt_ssh_cli(salt_master, salt_ssh_roster_file, ssh_keys, ssh_container, ssh_sub_container):
    if False:
        i = 10
        return i + 15
    assert salt_master.is_running()
    assert ssh_container.is_running()
    assert ssh_sub_container.is_running()
    return salt_master.salt_ssh_cli(timeout=180, roster_file=salt_ssh_roster_file, client_key=str(ssh_keys.priv_path), base_script_args=['--ignore-host-keys'])

@pytest.mark.flaky_jail
def test_setup(salt_ssh_cli, ssh_container_name, ssh_sub_container_name, ssh_password):
    if False:
        while True:
            i = 10
    '\n    Test salt-ssh setup works\n    '
    possible_ids = (ssh_container_name, ssh_sub_container_name)
    ret = salt_ssh_cli.run('--passwd', ssh_password, '--key-deploy', 'test.ping', minion_tgt='*')
    try:
        assert ret.returncode == 0
    except AssertionError:
        time.sleep(5)
        ret = salt_ssh_cli.run('--passwd', ssh_password, '--key-deploy', 'test.ping', minion_tgt='*')
        assert ret.returncode == 0
    for id in possible_ids:
        assert id in ret.data
        assert ret.data[id] is True
    ret = salt_ssh_cli.run('test.ping', minion_tgt='*')
    assert ret.returncode == 0
    for id in possible_ids:
        assert id in ret.data
        assert ret.data[id] is True
    sleep_time = 15
    cmdline = salt_ssh_cli.cmdline('test.sleep', sleep_time, minion_tgt='*')
    terminal_stdout = tempfile.SpooledTemporaryFile(512000, buffering=0)
    terminal_stderr = tempfile.SpooledTemporaryFile(512000, buffering=0)
    proc = subprocess.Popen(cmdline, shell=False, stdout=terminal_stdout, stderr=terminal_stderr, universal_newlines=True)
    start = time.time()
    try:
        proc.wait(1)
    except subprocess.TimeoutExpired:
        pass
    else:
        terminate_process(proc.pid, kill_children=True)
        pytest.fail('The test process failed to start')
    time.sleep(2)
    os.kill(proc.pid, signal.SIGINT)
    with proc:
        proc.wait(sleep_time * 2)
        proc.poll()
        proc.communicate()
    stop = time.time()
    terminal_stdout.flush()
    terminal_stdout.seek(0)
    stdout = proc._translate_newlines(terminal_stdout.read(), __salt_system_encoding__, sys.stdout.errors)
    terminal_stdout.close()
    terminal_stderr.flush()
    terminal_stderr.seek(0)
    stderr = proc._translate_newlines(terminal_stderr.read(), __salt_system_encoding__, sys.stderr.errors)
    terminal_stderr.close()
    ret = ProcessResult(returncode=proc.returncode, stdout=stdout, stderr=stderr, cmdline=proc.args)
    log.debug(ret)
    for id in possible_ids:
        assert id not in ret.stdout, "The command wasn't actually terminated. Took {} seconds.".format(round(stop - start, 2))