"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.modjk as modjk
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {modjk: {}}

def test_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for return the modjk version\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.jk_version': 'mod_jk/1.2.37'}):
        assert modjk.version() == '1.2.37'

def test_get_running():
    if False:
        return 10
    '\n    Test for get the current running config (not from disk)\n    '
    with patch.object(modjk, '_do_http', return_value={}):
        assert modjk.get_running() == {}

def test_dump_config():
    if False:
        i = 10
        return i + 15
    '\n    Test for dump the original configuration that was loaded from disk\n    '
    with patch.object(modjk, '_do_http', return_value={}):
        assert modjk.dump_config() == {}

def test_list_configured_members():
    if False:
        print('Hello World!')
    '\n    Test for return a list of member workers from the configuration files\n    '
    with patch.object(modjk, '_do_http', return_value={}):
        assert modjk.list_configured_members('loadbalancer1') == []
    with patch.object(modjk, '_do_http', return_value={'worker.loadbalancer1.balance_workers': 'SALT'}):
        assert modjk.list_configured_members('loadbalancer1') == ['SALT']

def test_workers():
    if False:
        print('Hello World!')
    '\n    Test for return a list of member workers and their status\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.list': 'Salt1,Salt2'}):
        assert modjk.workers() == {}

def test_recover_all():
    if False:
        return 10
    '\n    Test for set the all the workers in lbn to recover and\n    activate them if they are not\n    '
    with patch.object(modjk, '_do_http', return_value={}):
        assert modjk.recover_all('loadbalancer1') == {}
    with patch.object(modjk, '_do_http', return_value={'worker.loadbalancer1.balance_workers': 'SALT'}):
        with patch.object(modjk, 'worker_status', return_value={'activation': 'ACT', 'state': 'OK'}):
            assert modjk.recover_all('loadbalancer1') == {'SALT': {'activation': 'ACT', 'state': 'OK'}}

def test_reset_stats():
    if False:
        return 10
    '\n    Test for reset all runtime statistics for the load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.reset_stats('loadbalancer1')

def test_lb_edit():
    if False:
        while True:
            i = 10
    '\n    Test for edit the loadbalancer settings\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.lb_edit('loadbalancer1', {'vlr': 1, 'vlt': 60})

def test_bulk_stop():
    if False:
        print('Hello World!')
    '\n    Test for stop all the given workers in the specific load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.bulk_stop(['node1', 'node2', 'node3'], 'loadbalancer1')

def test_bulk_activate():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for activate all the given workers in the specific load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.bulk_activate(['node1', 'node2', 'node3'], 'loadbalancer1')

def test_bulk_disable():
    if False:
        i = 10
        return i + 15
    '\n    Test for disable all the given workers in the specific load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.bulk_disable(['node1', 'node2', 'node3'], 'loadbalancer1')

def test_bulk_recover():
    if False:
        i = 10
        return i + 15
    '\n    Test for recover all the given workers in the specific load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.bulk_recover(['node1', 'node2', 'node3'], 'loadbalancer1')

def test_worker_status():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for return the state of the worker\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.node1.activation': 'ACT', 'worker.node1.state': 'OK'}):
        assert modjk.worker_status('node1') == {'activation': 'ACT', 'state': 'OK'}
    with patch.object(modjk, '_do_http', return_value={}):
        assert not modjk.worker_status('node1')

def test_worker_recover():
    if False:
        i = 10
        return i + 15
    '\n    Test for set the worker to recover this module will fail\n    if it is in OK state\n    '
    with patch.object(modjk, '_do_http', return_value={}):
        assert modjk.worker_recover('node1', 'loadbalancer1') == {}

def test_worker_disable():
    if False:
        while True:
            i = 10
    '\n    Test for set the worker to disable state in the lbn load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.worker_disable('node1', 'loadbalancer1')

def test_worker_activate():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for set the worker to activate state in the lbn load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.worker_activate('node1', 'loadbalancer1')

def test_worker_stop():
    if False:
        i = 10
        return i + 15
    '\n    Test for set the worker to stopped state in the lbn load balancer\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.worker_stop('node1', 'loadbalancer1')

def test_worker_edit():
    if False:
        return 10
    '\n    Test for edit the worker settings\n    '
    with patch.object(modjk, '_do_http', return_value={'worker.result.type': 'OK'}):
        assert modjk.worker_edit('node1', 'loadbalancer1', {'vwf': 500, 'vwd': 60})