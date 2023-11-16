import os
from unittest import mock
import pytest
from lightning.fabric.plugins.environments import LSFEnvironment

def _make_rankfile(tmp_path):
    if False:
        print('Hello World!')
    hosts = 'batch\n10.10.10.0\n10.10.10.1\n10.10.10.2\n10.10.10.3'
    p = tmp_path / 'lsb_djob_rankfile'
    p.write_text(hosts)
    return str(p)

@mock.patch.dict(os.environ, {'LSB_JOBID': '1234'})
def test_missing_lsb_djob_rankfile():
    if False:
        print('Hello World!')
    'Test an error when the LSB_DJOB_RANKFILE cannot be found.'
    with pytest.raises(ValueError, match='Did not find the environment variable `LSB_DJOB_RANKFILE`'):
        LSFEnvironment()

@mock.patch.dict(os.environ, {'LSB_DJOB_RANKFILE': '', 'LSB_JOBID': '1234'})
def test_empty_lsb_djob_rankfile():
    if False:
        for i in range(10):
            print('nop')
    'Test an error when the LSB_DJOB_RANKFILE is not populated.'
    with pytest.raises(ValueError, match='The environment variable `LSB_DJOB_RANKFILE` is empty'):
        LSFEnvironment()

def test_missing_lsb_job_id(tmp_path):
    if False:
        while True:
            i = 10
    'Test an error when the job id cannot be found.'
    with mock.patch.dict(os.environ, {'LSB_DJOB_RANKFILE': _make_rankfile(tmp_path)}), pytest.raises(ValueError, match='Could not find job id in environment variable LSB_JOBID'):
        LSFEnvironment()

def test_manual_main_port_and_address(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test a user can set the port manually through the MASTER_PORT env variable.'
    environ = {'LSB_DJOB_RANKFILE': _make_rankfile(tmp_path), 'LSB_JOBID': '1234', 'JSM_NAMESPACE_SIZE': '4', 'JSM_NAMESPACE_RANK': '3', 'JSM_NAMESPACE_LOCAL_RANK': '1'}
    with mock.patch.dict(os.environ, environ), mock.patch('socket.gethostname', return_value='10.10.10.2'):
        env = LSFEnvironment()
        assert env.main_port == 10234

def test_attributes_from_environment_variables(tmp_path):
    if False:
        return 10
    'Test that the LSF environment takes the attributes from the environment variables.'
    environ = {'LSB_DJOB_RANKFILE': _make_rankfile(tmp_path), 'LSB_JOBID': '1234', 'JSM_NAMESPACE_SIZE': '4', 'JSM_NAMESPACE_RANK': '3', 'JSM_NAMESPACE_LOCAL_RANK': '1'}
    with mock.patch.dict(os.environ, environ), mock.patch('socket.gethostname', return_value='10.10.10.2'):
        env = LSFEnvironment()
        assert env.creates_processes_externally
        assert env.main_address == '10.10.10.0'
        assert env.main_port == 10234
        assert env.world_size() == 4
        assert env.global_rank() == 3
        assert env.local_rank() == 1
        env.set_global_rank(100)
        assert env.global_rank() == 3
        env.set_world_size(100)
        assert env.world_size() == 4
        assert LSFEnvironment.detect()

def test_node_rank(tmp_path):
    if False:
        print('Hello World!')
    environ = {'LSB_DJOB_RANKFILE': _make_rankfile(tmp_path), 'LSB_JOBID': '1234', 'JSM_NAMESPACE_SIZE': '4', 'JSM_NAMESPACE_RANK': '3', 'JSM_NAMESPACE_LOCAL_RANK': '1'}
    with mock.patch.dict(os.environ, environ), mock.patch('socket.gethostname', return_value='10.10.10.2'):
        env = LSFEnvironment()
        assert env.node_rank() == 2

def test_detect():
    if False:
        while True:
            i = 10
    'Test the detection of a LSF environment configuration.'
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not LSFEnvironment.detect()
    with mock.patch.dict(os.environ, {'LSB_DJOB_RANKFILE': '', 'LSB_JOBID': '', 'JSM_NAMESPACE_SIZE': '', 'JSM_NAMESPACE_LOCAL_RANK': ''}):
        assert LSFEnvironment.detect()