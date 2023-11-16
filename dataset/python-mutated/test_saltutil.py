import pytest
import salt.config
import salt.loader
import salt.modules.saltutil
import salt.state
from tests.support.mock import patch

@pytest.fixture
def opts(salt_master_factory):
    if False:
        i = 10
        return i + 15
    config_overrides = {'master_uri': 'tcp://127.0.0.1:11111'}
    factory = salt_master_factory.salt_minion_daemon('get-tops-minion', overrides=config_overrides)
    yield factory.config.copy()

@pytest.fixture
def modules(opts):
    if False:
        while True:
            i = 10
    yield salt.loader.minion_mods(opts, context={})

@pytest.fixture
def configure_mocks(opts):
    if False:
        print('Hello World!')
    with patch('salt.utils.extmods.sync', return_value=(None, None)):
        with patch.object(salt.state.HighState, 'top_matches', return_value={}):
            with patch.object(salt.state.BaseHighState, '_BaseHighState__gen_opts', return_value=opts):
                with patch.object(salt.state.State, '_gather_pillar', return_value={}):
                    yield

@pytest.fixture
def destroy(configure_mocks):
    if False:
        while True:
            i = 10
    with patch.object(salt.state.HighState, 'destroy') as destroy:
        yield destroy

@pytest.fixture
def get_top(configure_mocks):
    if False:
        i = 10
        return i + 15
    with patch.object(salt.state.HighState, 'get_top') as get_top:
        yield get_top

@pytest.mark.slow_test
def test__get_top_file_envs(modules, get_top, destroy):
    if False:
        i = 10
        return i + 15
    '\n    Ensure we cleanup objects created by saltutil._get_top_file_envs #60449\n    '
    modules['saltutil.sync_clouds']()
    assert get_top.called
    assert destroy.called