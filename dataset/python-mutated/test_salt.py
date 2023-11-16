"""
Tests using salt formula
"""
import pytest

@pytest.fixture(scope='module')
def _formula(saltstack_formula):
    if False:
        print('Hello World!')
    with saltstack_formula(name='salt-formula', tag='1.12.0') as formula:
        yield formula

@pytest.fixture(scope='module')
def modules(loaders, _formula):
    if False:
        i = 10
        return i + 15
    return loaders.modules

@pytest.mark.skip_on_windows
@pytest.mark.destructive_test
def test_salt_formula(modules):
    if False:
        while True:
            i = 10
    ret = modules.state.sls('salt.master')
    for staterun in ret:
        assert not staterun.result.failed
    ret = modules.state.sls('salt.minion')
    for staterun in ret:
        assert not staterun.result.failed