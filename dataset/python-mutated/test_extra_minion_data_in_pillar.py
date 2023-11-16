import pytest
from salt.pillar import extra_minion_data_in_pillar
from tests.support.mock import MagicMock

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {extra_minion_data_in_pillar: {}}

@pytest.fixture
def extra_minion_data():
    if False:
        print('Hello World!')
    return {'key1': {'subkey1': 'value1'}, 'key2': {'subkey2': {'subsubkey2': 'value2'}}, 'key3': 'value3', 'key4': {'subkey4': 'value4'}}

def test_extra_values_none_or_empty():
    if False:
        print('Hello World!')
    ret = extra_minion_data_in_pillar.ext_pillar('fake_id', MagicMock(), 'fake_include', None)
    assert ret == {}
    ret = extra_minion_data_in_pillar.ext_pillar('fake_id', MagicMock(), 'fake_include', {})
    assert ret == {}

def test_include_all(extra_minion_data):
    if False:
        for i in range(10):
            print('nop')
    for include_all in ['*', '<all>']:
        ret = extra_minion_data_in_pillar.ext_pillar('fake_id', MagicMock(), include_all, extra_minion_data)
        assert ret == extra_minion_data

def test_include_specific_keys(extra_minion_data):
    if False:
        return 10
    ret = extra_minion_data_in_pillar.ext_pillar('fake_id', MagicMock(), include=['key1:subkey1', 'key2:subkey3', 'key3', 'key4'], extra_minion_data=extra_minion_data)
    assert ret == {'key1': {'subkey1': 'value1'}, 'key3': 'value3', 'key4': {'subkey4': 'value4'}}