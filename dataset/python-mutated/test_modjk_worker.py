"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.states.modjk_worker as modjk_worker
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {modjk_worker: {}}

def test_stop():
    if False:
        while True:
            i = 10
    '\n    Test to stop the named worker from the lbn load balancers\n     at the targeted minions.\n    '
    name = "{{ grains['id'] }}"
    lbn = 'application'
    target = 'roles:balancer'
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    comt = 'no servers answered the published command modjk.worker_status'
    mock = MagicMock(return_value=False)
    with patch.dict(modjk_worker.__salt__, {'publish.publish': mock}):
        ret.update({'comment': comt})
        assert modjk_worker.stop(name, lbn, target) == ret

def test_activate():
    if False:
        i = 10
        return i + 15
    '\n    Test to activate the named worker from the lbn load balancers\n     at the targeted minions.\n    '
    name = "{{ grains['id'] }}"
    lbn = 'application'
    target = 'roles:balancer'
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    comt = 'no servers answered the published command modjk.worker_status'
    mock = MagicMock(return_value=False)
    with patch.dict(modjk_worker.__salt__, {'publish.publish': mock}):
        ret.update({'comment': comt})
        assert modjk_worker.activate(name, lbn, target) == ret

def test_disable():
    if False:
        return 10
    '\n    Test to disable the named worker from the lbn load balancers\n     at the targeted minions.\n    '
    name = "{{ grains['id'] }}"
    lbn = 'application'
    target = 'roles:balancer'
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    comt = 'no servers answered the published command modjk.worker_status'
    mock = MagicMock(return_value=False)
    with patch.dict(modjk_worker.__salt__, {'publish.publish': mock}):
        ret.update({'comment': comt})
        assert modjk_worker.disable(name, lbn, target) == ret