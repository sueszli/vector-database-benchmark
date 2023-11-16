"""
Unit tests for salt.auth.ldap
"""
import pytest
import salt.auth.ldap
from tests.support.mock import patch
pytestmark = [pytest.mark.skipif(not salt.auth.ldap.HAS_LDAP, reason='Install python-ldap for this test')]

class Bind:
    """
    fake search_s return
    """

    @staticmethod
    def search_s(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return [('cn=saltusers,cn=groups,cn=compat,dc=saltstack,dc=com', {'memberUid': [b'saltuser'], 'cn': [b'saltusers']})]

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {salt.auth.ldap: {'__opts__': {'auth.ldap.binddn': 'uid={{username}},cn=users,cn=compat,dc=saltstack,dc=com', 'auth.ldap.port': 389, 'auth.ldap.tls': False, 'auth.ldap.server': '172.18.0.2', 'auth.ldap.accountattributename': 'memberUid', 'auth.ldap.groupattribute': 'memberOf', 'auth.ldap.group_basedn': 'cn=groups,cn=compat,dc=saltstack,dc=com', 'auth.ldap.basedn': 'dc=saltstack,dc=com', 'auth.ldap.group_filter': '(&(memberUid={{ username }})(objectClass=posixgroup))'}}}

def test_config():
    if False:
        i = 10
        return i + 15
    '\n    Test that the _config function works correctly\n    '
    assert salt.auth.ldap._config('basedn') == 'dc=saltstack,dc=com'
    assert salt.auth.ldap._config('group_filter') == '(&(memberUid={{ username }})(objectClass=posixgroup))'
    assert salt.auth.ldap._config('accountattributename') == 'memberUid'
    assert salt.auth.ldap._config('groupattribute') == 'memberOf'

def test_groups_freeipa():
    if False:
        return 10
    '\n    test groups in freeipa\n    '
    with patch.dict(salt.auth.ldap.__opts__, {'auth.ldap.freeipa': True}):
        with patch('salt.auth.ldap._bind', return_value=Bind):
            assert 'saltusers' in salt.auth.ldap.groups('saltuser', password='password')

def test_groups():
    if False:
        i = 10
        return i + 15
    '\n    test groups in ldap\n    '
    with patch('salt.auth.ldap._bind', return_value=Bind):
        assert 'saltusers' in salt.auth.ldap.groups('saltuser', password='password')

def test_groups_activedirectory():
    if False:
        return 10
    '\n    test groups in activedirectory\n    '
    with patch.dict(salt.auth.ldap.__opts__, {'auth.ldap.activedirectory': True}):
        with patch('salt.auth.ldap._bind', return_value=Bind):
            assert 'saltusers' in salt.auth.ldap.groups('saltuser', password='password')

def test_auth_nopass():
    if False:
        return 10
    with patch.dict(salt.auth.ldap.__opts__, {'auth.ldap.bindpw': 'p@ssw0rd!'}):
        with patch('salt.auth.ldap._bind_for_search', return_value=Bind):
            assert not salt.auth.ldap.auth('foo', None)

def test_auth_nouser():
    if False:
        print('Hello World!')
    with patch.dict(salt.auth.ldap.__opts__, {'auth.ldap.bindpw': 'p@ssw0rd!'}):
        with patch('salt.auth.ldap._bind_for_search', return_value=Bind):
            assert not salt.auth.ldap.auth(None, 'foo')

def test_auth_nouserandpass():
    if False:
        i = 10
        return i + 15
    with patch.dict(salt.auth.ldap.__opts__, {'auth.ldap.bindpw': 'p@ssw0rd!'}):
        with patch('salt.auth.ldap._bind_for_search', return_value=Bind):
            assert not salt.auth.ldap.auth(None, None)