"""
    :codeauthor: Mike Wiebe <@mikewiebe>
"""
import pytest
import salt.states.nxos as nxos_state
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {nxos_state: {}}

def test_user_present_create():
    if False:
        for i in range(10):
            print('nop')
    '\n    user_present method - create\n    '
    roles = ['vdc-admin']
    salt_mock = {'nxos.get_roles': MagicMock(side_effect=[[], roles, roles]), 'nxos.get_user': MagicMock(side_effect=['']), 'nxos.set_role': MagicMock(side_effect=['set_role'])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_present('daniel', roles=roles)
            assert result['name'] == 'daniel'
            assert result['result']
            assert result['changes']['roles']['new'] == ['vdc-admin']
            assert result['changes']['roles']['old'] == []
            assert result['comment'] == 'User set correctly'

def test_user_present_create_opts_test():
    if False:
        while True:
            i = 10
    '\n    user_present method - create opts\n    '
    roles = ['vdc-admin']
    salt_mock = {'nxos.get_roles': MagicMock(side_effect=[[], roles, roles]), 'nxos.get_user': MagicMock(side_effect=['']), 'nxos.set_role': MagicMock(side_effect=['set_role'])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_present('daniel', roles=roles)
            assert result['name'] == 'daniel'
            assert result['result'] is None
            assert result['changes']['role']['add'] == ['vdc-admin']
            assert result['changes']['role']['remove'] == []
            assert result['comment'] == 'User will be created'

def test_user_present_create_non_defaults():
    if False:
        while True:
            i = 10
    '\n    user_present method - create non default opts\n    '
    username = 'daniel'
    password = 'ghI&435y55#'
    roles = ['vdc-admin', 'dev-ops']
    encrypted = False
    crypt_salt = 'foobar123'
    algorithm = 'md5'
    salt_mock = {'nxos.check_password': MagicMock(side_effect=[False, True]), 'nxos.get_roles': MagicMock(side_effect=[[], roles, roles]), 'nxos.get_user': MagicMock(side_effect=['']), 'nxos.set_password': MagicMock(side_effect=['new_user']), 'nxos.set_role': MagicMock(side_effect=['set_role', 'set_role'])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_present(username, password=password, roles=roles, encrypted=encrypted, crypt_salt=crypt_salt, algorithm=algorithm)
            assert result['name'] == 'daniel'
            assert result['result']
            assert result['changes']['password']['new'] == 'new_user'
            assert result['changes']['password']['old'] == ''
            assert result['changes']['roles']['new'] == ['vdc-admin', 'dev-ops']
            assert result['changes']['roles']['old'] == []
            assert result['comment'] == 'User set correctly'

def test_user_present_create_encrypted_password_no_roles_opts_test():
    if False:
        i = 10
        return i + 15
    '\n    user_present method - encrypted password, no roles\n    '
    username = 'daniel'
    password = '$1$foobar12$K7x4Rxua11qakvrRjcwDC/'
    encrypted = True
    crypt_salt = 'foobar123'
    algorithm = 'md5'
    salt_mock = {'nxos.check_password': MagicMock(side_effect=[False, True]), 'nxos.get_user': MagicMock(side_effect=['']), 'nxos.set_password': MagicMock(side_effect=['new_user'])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_present(username, password=password, encrypted=encrypted, crypt_salt=crypt_salt, algorithm=algorithm)
            assert result['name'] == 'daniel'
            assert result['result'] is None
            assert result['changes']['password'] is True
            assert result['comment'] == 'User will be created'

def test_user_present_create_user_exists():
    if False:
        while True:
            i = 10
    '\n    user_present method - user exists\n    '
    username = 'daniel'
    password = '$1$foobar12$K7x4Rxua11qakvrRjcwDC/'
    encrypted = True
    crypt_salt = 'foobar123'
    algorithm = 'md5'
    salt_mock = {'nxos.check_password': MagicMock(side_effect=[True]), 'nxos.get_user': MagicMock(side_effect=['user_exists'])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_present(username, password=password, encrypted=encrypted, crypt_salt=crypt_salt, algorithm=algorithm)
            assert result['name'] == 'daniel'
            assert result['result']
            assert result['changes'] == {}
            assert result['comment'] == 'User already exists'

def test_user_present_create_user_exists_opts_test():
    if False:
        while True:
            i = 10
    '\n    user_present method - user exists with opts\n    '
    username = 'daniel'
    password = '$1$foobar12$K7x4Rxua11qakvrRjcwDC/'
    roles = ['vdc-admin', 'dev-opts']
    new_roles = ['network-operator']
    encrypted = True
    crypt_salt = 'foobar123'
    algorithm = 'md5'
    salt_mock = {'nxos.check_password': MagicMock(side_effect=[True]), 'nxos.get_roles': MagicMock(side_effect=[roles]), 'nxos.get_user': MagicMock(side_effect=['user_exists'])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_present(username, password=password, roles=new_roles, encrypted=encrypted, crypt_salt=crypt_salt, algorithm=algorithm)
            remove = result['changes']['roles']['remove']
            remove.sort()
            assert result['name'] == 'daniel'
            assert result['result'] is None
            assert result['changes']['roles']['add'] == ['network-operator']
            assert remove == ['dev-opts', 'vdc-admin']
            assert result['comment'] == 'User will be updated'

def test_user_absent():
    if False:
        while True:
            i = 10
    '\n    user_absent method - remove user\n    '
    username = 'daniel'
    salt_mock = {'nxos.get_user': MagicMock(side_effect=['daniel', '']), 'nxos.remove_user': MagicMock(side_effect=['remove_user'])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_absent(username)
            assert result['name'] == 'daniel'
            assert result['result']
            assert result['changes']['old'] == 'daniel'
            assert result['changes']['new'] == ''
            assert result['comment'] == 'User removed'

def test_user_absent_user_does_not_exist():
    if False:
        while True:
            i = 10
    '\n    user_absent method - remove user\n    '
    username = 'daniel'
    side_effect = MagicMock(side_effect=[''])
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, {'nxos.get_user': side_effect}):
            result = nxos_state.user_absent(username)
            assert result['name'] == 'daniel'
            assert result['result']
            assert result['changes'] == {}
            assert result['comment'] == 'User does not exist'

def test_user_absent_test_opts():
    if False:
        for i in range(10):
            print('nop')
    '\n    user_absent method - remove user with opts\n    '
    username = 'daniel'
    salt_mock = {'nxos.get_user': MagicMock(side_effect=['daniel', '']), 'nxos.remove_user': MagicMock(side_effect=['remove_user'])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.user_absent(username)
            assert result['name'] == 'daniel'
            assert result['result'] is None
            assert result['changes']['old'] == 'daniel'
            assert result['changes']['new'] == ''
            assert result['comment'] == 'User will be removed'

def test_config_present():
    if False:
        i = 10
        return i + 15
    '\n    config_present method - add config\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    snmp_matches1 = ['snmp-server community randomSNMPstringHERE group network-operator']
    snmp_matches2 = [['snmp-server community AnotherRandomSNMPSTring group network-admin']]
    salt_mock = {'nxos.config': MagicMock(side_effect=['add_snmp_config1', 'add_snmp_config2']), 'nxos.find': MagicMock(side_effect=[[], snmp_matches1, snmp_matches2])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.config_present(config_data)
            assert result['name'] == config_data
            assert result['result']
            assert result['changes']['new'] == config_data
            assert result['comment'] == 'Successfully added config'

def test_config_present_already_configured():
    if False:
        for i in range(10):
            print('nop')
    '\n    config_present method - add config already configured\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    side_effect = MagicMock(side_effect=[config_data[0], config_data[1]])
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, {'nxos.find': side_effect}):
            result = nxos_state.config_present(config_data)
            assert result['name'] == config_data
            assert result['result']
            assert result['changes'] == {}
            assert result['comment'] == 'Config is already set'

def test_config_present_test_opts():
    if False:
        while True:
            i = 10
    '\n    config_present method - add config\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    snmp_matches1 = ['snmp-server community randomSNMPstringHERE group network-operator']
    snmp_matches2 = [['snmp-server community AnotherRandomSNMPSTring group network-admin']]
    salt_mock = {'nxos.config': MagicMock(side_effect=['add_snmp_config1', 'add_snmp_config2']), 'nxos.find': MagicMock(side_effect=[[], snmp_matches1, snmp_matches2])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.config_present(config_data)
            assert result['name'] == config_data
            assert result['result'] is None
            assert result['changes']['new'] == config_data
            assert result['comment'] == 'Config will be added'

def test_config_present_fail_to_add():
    if False:
        print('Hello World!')
    '\n    config_present method - add config fails\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    snmp_matches1 = ['snmp-server community randomSNMPstringHERE group network-operator']
    snmp_matches2 = [['snmp-server community AnotherRandomSNMPSTring group network-admin']]
    salt_mock = {'nxos.config': MagicMock(side_effect=['add_snmp_config1', 'add_snmp_config2']), 'nxos.find': MagicMock(side_effect=[[], '', ''])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.config_present(config_data)
            assert result['name'] == config_data
            assert not result['result']
            assert result['changes'] == {}
            assert result['comment'] == 'Failed to add config'

def test_replace():
    if False:
        for i in range(10):
            print('nop')
    '\n    replace method - replace config\n    '
    name = 'randomSNMPstringHERE'
    repl = 'NEWrandoSNMPstringHERE'
    matches_before = ['snmp-server community randomSNMPstringHERE group network-operator']
    match_after = []
    changes = {}
    changes['new'] = ['snmp-server community NEWrandoSNMPstringHERE group network-operator']
    changes['old'] = ['snmp-server community randomSNMPstringHERE group network-operator']
    salt_mock = {'nxos.find': MagicMock(side_effect=[matches_before, match_after]), 'nxos.replace': MagicMock(side_effect=[changes])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.replace(name, repl)
            assert result['name'] == name
            assert result['result']
            assert result['changes']['new'] == changes['new']
            assert result['changes']['old'] == changes['old']
            assert result['comment'] == 'Successfully replaced all instances of "randomSNMPstringHERE" with "NEWrandoSNMPstringHERE"'

def test_replace_test_opts():
    if False:
        return 10
    '\n    replace method - replace config\n    '
    name = 'randomSNMPstringHERE'
    repl = 'NEWrandoSNMPstringHERE'
    matches_before = ['snmp-server community randomSNMPstringHERE group network-operator']
    match_after = []
    changes = {}
    changes['new'] = ['snmp-server community NEWrandoSNMPstringHERE group network-operator']
    changes['old'] = ['snmp-server community randomSNMPstringHERE group network-operator']
    salt_mock = {'nxos.find': MagicMock(side_effect=[matches_before, match_after]), 'nxos.replace': MagicMock(side_effect=[changes])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.replace(name, repl)
            assert result['name'] == name
            assert result['result'] is None
            assert result['changes']['new'] == changes['new']
            assert result['changes']['old'] == changes['old']
            assert result['comment'] == 'Configs will be changed'

def test_config_absent():
    if False:
        i = 10
        return i + 15
    '\n    config_absent method - remove config\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    snmp_matches1 = ['snmp-server community randomSNMPstringHERE group network-operator']
    snmp_matches2 = [['snmp-server community AnotherRandomSNMPSTring group network-admin']]
    salt_mock = {'nxos.delete_config': MagicMock(side_effect=['remove_config', 'remove_config']), 'nxos.find': MagicMock(side_effect=[snmp_matches1, [], snmp_matches2, []])}
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.config_absent(config_data)
            assert result['name'] == config_data
            assert result['result']
            assert result['changes']['new'] == config_data
            assert result['comment'] == 'Successfully deleted config'

def test_config_absent_already_configured():
    if False:
        while True:
            i = 10
    '\n    config_absent method - add config removed\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    side_effect = MagicMock(side_effect=[[], []])
    with patch.dict(nxos_state.__opts__, {'test': False}):
        with patch.dict(nxos_state.__salt__, {'nxos.find': side_effect}):
            result = nxos_state.config_absent(config_data)
            assert result['name'] == config_data
            assert result['result']
            assert result['changes'] == {}
            assert result['comment'] == 'Config is already absent'

def test_config_absent_test_opts():
    if False:
        for i in range(10):
            print('nop')
    '\n    config_absent method - remove config\n    '
    config_data = ['snmp-server community randomSNMPstringHERE group network-operator', 'snmp-server community AnotherRandomSNMPSTring group network-admin']
    snmp_matches1 = ['snmp-server community randomSNMPstringHERE group network-operator']
    snmp_matches2 = [['snmp-server community AnotherRandomSNMPSTring group network-admin']]
    salt_mock = {'nxos.delete_config': MagicMock(side_effect=['remove_config', 'remove_config']), 'nxos.find': MagicMock(side_effect=[snmp_matches1, [], snmp_matches2, []])}
    with patch.dict(nxos_state.__opts__, {'test': True}):
        with patch.dict(nxos_state.__salt__, salt_mock):
            result = nxos_state.config_absent(config_data)
            assert result['name'] == config_data
            assert result['result'] is None
            assert result['changes']['new'] == config_data
            assert result['comment'] == 'Config will be removed'