import pytest
import salt.modules.baredoc as baredoc
from tests.support.paths import SALT_CODE_DIR

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {baredoc: {'__opts__': {'extension_modules': SALT_CODE_DIR}, '__grains__': {'saltpath': SALT_CODE_DIR}}}

def test_baredoc_list_states():
    if False:
        i = 10
        return i + 15
    '\n    Test baredoc state module listing\n    '
    ret = baredoc.list_states(names_only=True)
    assert 'value_present' in ret['xml'][0]

def test_baredoc_list_states_args():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test baredoc state listing with args\n    '
    ret = baredoc.list_states()
    assert 'value_present' in ret['xml'][0]
    assert 'xpath' in ret['xml'][0]['value_present']

def test_baredoc_list_states_single():
    if False:
        i = 10
        return i + 15
    '\n    Test baredoc state listing single state module\n    '
    ret = baredoc.list_states('xml')
    assert 'value_present' in ret['xml'][0]
    assert 'xpath' in ret['xml'][0]['value_present']

def test_baredoc_list_modules():
    if False:
        while True:
            i = 10
    '\n    test baredoc executiion module listing\n    '
    ret = baredoc.list_modules(names_only=True)
    assert 'get_value' in ret['xml'][0]

def test_baredoc_list_modules_args():
    if False:
        return 10
    '\n    test baredoc execution module listing with args\n    '
    ret = baredoc.list_modules()
    assert 'get_value' in ret['xml'][0]
    assert 'file' in ret['xml'][0]['get_value']

def test_baredoc_list_modules_single_and_alias():
    if False:
        i = 10
        return i + 15
    '\n    test baredoc single module listing\n    '
    ret = baredoc.list_modules('mdata')
    assert 'put' in ret['mdata'][2]
    assert 'keyname' in ret['mdata'][2]['put']

def test_baredoc_state_docs():
    if False:
        for i in range(10):
            print('nop')
    ret = baredoc.state_docs()
    assert 'XML Manager' in ret['xml']
    assert 'zabbix_usergroup' in ret

def test_baredoc_state_docs_single_arg():
    if False:
        i = 10
        return i + 15
    ret = baredoc.state_docs('xml')
    assert 'XML Manager' in ret['xml']
    ret = baredoc.state_docs('xml.value_present')
    assert 'Manages a given XML file' in ret['xml.value_present']

def test_baredoc_state_docs_multiple_args():
    if False:
        i = 10
        return i + 15
    ret = baredoc.state_docs('zabbix_hostgroup.present', 'xml')
    assert 'Ensures that the host group exists' in ret['zabbix_hostgroup.present']
    assert 'XML Manager' in ret['xml']
    assert 'Manages a given XML file' in ret['xml.value_present']

def test_baredoc_module_docs():
    if False:
        for i in range(10):
            print('nop')
    ret = baredoc.module_docs()
    assert 'A module for testing' in ret['saltcheck']

def test_baredoc_module_docs_single_arg():
    if False:
        for i in range(10):
            print('nop')
    ret = baredoc.module_docs('saltcheck')
    assert 'A module for testing' in ret['saltcheck']

def test_baredoc_module_docs_multiple_args():
    if False:
        for i in range(10):
            print('nop')
    ret = baredoc.module_docs('saltcheck', 'xml.get_value')
    assert 'A module for testing' in ret['saltcheck']
    assert 'Returns the value of the matched xpath element' in ret['xml.get_value']