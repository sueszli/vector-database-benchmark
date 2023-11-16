import pytest
import salt.config
import salt.daemons.masterapi as masterapi
import salt.utils.platform
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.slow_test]

@pytest.fixture
def check_keys():
    if False:
        i = 10
        return i + 15
    return {'test': 'mGXdurU1c8lXt5cmpbGq4rWvrOvDXxkwI9gbkP5CBBjpyGWuB8vkgz9r+sjjG0wVDL9/uFuREtk=', 'root': '2t5HHv/ek2wIFh8tTX2c3hdt+6V+93xKlcXb7IlGLIszOeCVv2NuH38LyCw9UwQTfUFTeseXhSs='}

@pytest.fixture
def local_funcs(master_opts):
    if False:
        return 10
    opts = salt.config.master_config(None)
    return masterapi.LocalFuncs(opts, 'test-key')

@pytest.fixture
def check_local_funcs(master_opts, check_keys):
    if False:
        print('Hello World!')
    return masterapi.LocalFuncs(master_opts, check_keys)

def test_runner_token_not_authenticated(local_funcs):
    if False:
        print('Hello World!')
    "\n    Asserts that a TokenAuthenticationError is returned when the token can't authenticate.\n    "
    mock_ret = {'error': {'name': 'TokenAuthenticationError', 'message': 'Authentication failure of type "token" occurred.'}}
    ret = local_funcs.runner({'token': 'asdfasdfasdfasdf'})
    assert mock_ret == ret

def test_runner_token_authorization_error(local_funcs):
    if False:
        i = 10
        return i + 15
    '\n    Asserts that a TokenAuthenticationError is returned when the token authenticates, but is\n    not authorized.\n    '
    token = 'asdfasdfasdfasdf'
    load = {'token': token, 'fun': 'test.arg', 'kwarg': {}}
    mock_token = {'token': token, 'eauth': 'foo', 'name': 'test'}
    mock_ret = {'error': {'name': 'TokenAuthenticationError', 'message': 'Authentication failure of type "token" occurred for user test.'}}
    with patch('salt.auth.LoadAuth.authenticate_token', MagicMock(return_value=mock_token)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=[])):
        ret = local_funcs.runner(load)
    assert mock_ret == ret

def test_runner_token_salt_invocation_error(local_funcs):
    if False:
        return 10
    '\n    Asserts that a SaltInvocationError is returned when the token authenticates, but the\n    command is malformed.\n    '
    token = 'asdfasdfasdfasdf'
    load = {'token': token, 'fun': 'badtestarg', 'kwarg': {}}
    mock_token = {'token': token, 'eauth': 'foo', 'name': 'test'}
    mock_ret = {'error': {'name': 'SaltInvocationError', 'message': 'A command invocation error occurred: Check syntax.'}}
    with patch('salt.auth.LoadAuth.authenticate_token', MagicMock(return_value=mock_token)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=['testing'])):
        ret = local_funcs.runner(load)
    assert mock_ret == ret

def test_runner_eauth_not_authenticated(local_funcs):
    if False:
        print('Hello World!')
    "\n    Asserts that an EauthAuthenticationError is returned when the user can't authenticate.\n    "
    mock_ret = {'error': {'name': 'EauthAuthenticationError', 'message': 'Authentication failure of type "eauth" occurred for user UNKNOWN.'}}
    ret = local_funcs.runner({'eauth': 'foo'})
    assert mock_ret == ret

def test_runner_eauth_authorization_error(local_funcs):
    if False:
        return 10
    '\n    Asserts that an EauthAuthenticationError is returned when the user authenticates, but is\n    not authorized.\n    '
    load = {'eauth': 'foo', 'username': 'test', 'fun': 'test.arg', 'kwarg': {}}
    mock_ret = {'error': {'name': 'EauthAuthenticationError', 'message': 'Authentication failure of type "eauth" occurred for user test.'}}
    with patch('salt.auth.LoadAuth.authenticate_eauth', MagicMock(return_value=True)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=[])):
        ret = local_funcs.runner(load)
    assert mock_ret == ret

def test_runner_eauth_salt_invocation_error(local_funcs):
    if False:
        return 10
    '\n    Asserts that an EauthAuthenticationError is returned when the user authenticates, but the\n    command is malformed.\n    '
    load = {'eauth': 'foo', 'username': 'test', 'fun': 'bad.test.arg.func', 'kwarg': {}}
    mock_ret = {'error': {'name': 'SaltInvocationError', 'message': 'A command invocation error occurred: Check syntax.'}}
    with patch('salt.auth.LoadAuth.authenticate_eauth', MagicMock(return_value=True)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=['testing'])):
        ret = local_funcs.runner(load)
    assert mock_ret == ret

def test_wheel_token_not_authenticated(local_funcs):
    if False:
        while True:
            i = 10
    "\n    Asserts that a TokenAuthenticationError is returned when the token can't authenticate.\n    "
    mock_ret = {'error': {'name': 'TokenAuthenticationError', 'message': 'Authentication failure of type "token" occurred.'}}
    ret = local_funcs.wheel({'token': 'asdfasdfasdfasdf'})
    assert mock_ret == ret

def test_wheel_token_authorization_error(local_funcs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts that a TokenAuthenticationError is returned when the token authenticates, but is\n    not authorized.\n    '
    token = 'asdfasdfasdfasdf'
    load = {'token': token, 'fun': 'test.arg', 'kwarg': {}}
    mock_token = {'token': token, 'eauth': 'foo', 'name': 'test'}
    mock_ret = {'error': {'name': 'TokenAuthenticationError', 'message': 'Authentication failure of type "token" occurred for user test.'}}
    with patch('salt.auth.LoadAuth.authenticate_token', MagicMock(return_value=mock_token)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=[])):
        ret = local_funcs.wheel(load)
    assert mock_ret == ret

def test_wheel_token_salt_invocation_error(local_funcs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts that a SaltInvocationError is returned when the token authenticates, but the\n    command is malformed.\n    '
    token = 'asdfasdfasdfasdf'
    load = {'token': token, 'fun': 'badtestarg', 'kwarg': {}}
    mock_token = {'token': token, 'eauth': 'foo', 'name': 'test'}
    mock_ret = {'error': {'name': 'SaltInvocationError', 'message': 'A command invocation error occurred: Check syntax.'}}
    with patch('salt.auth.LoadAuth.authenticate_token', MagicMock(return_value=mock_token)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=['testing'])):
        ret = local_funcs.wheel(load)
    assert mock_ret == ret

def test_wheel_eauth_not_authenticated(local_funcs):
    if False:
        i = 10
        return i + 15
    "\n    Asserts that an EauthAuthenticationError is returned when the user can't authenticate.\n    "
    mock_ret = {'error': {'name': 'EauthAuthenticationError', 'message': 'Authentication failure of type "eauth" occurred for user UNKNOWN.'}}
    ret = local_funcs.wheel({'eauth': 'foo'})
    assert mock_ret == ret

def test_wheel_eauth_authorization_error(local_funcs):
    if False:
        while True:
            i = 10
    '\n    Asserts that an EauthAuthenticationError is returned when the user authenticates, but is\n    not authorized.\n    '
    load = {'eauth': 'foo', 'username': 'test', 'fun': 'test.arg', 'kwarg': {}}
    mock_ret = {'error': {'name': 'EauthAuthenticationError', 'message': 'Authentication failure of type "eauth" occurred for user test.'}}
    with patch('salt.auth.LoadAuth.authenticate_eauth', MagicMock(return_value=True)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=[])):
        ret = local_funcs.wheel(load)
    assert mock_ret == ret

def test_wheel_eauth_salt_invocation_error(local_funcs):
    if False:
        return 10
    '\n    Asserts that an EauthAuthenticationError is returned when the user authenticates, but the\n    command is malformed.\n    '
    load = {'eauth': 'foo', 'username': 'test', 'fun': 'bad.test.arg.func', 'kwarg': {}}
    mock_ret = {'error': {'name': 'SaltInvocationError', 'message': 'A command invocation error occurred: Check syntax.'}}
    with patch('salt.auth.LoadAuth.authenticate_eauth', MagicMock(return_value=True)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=['testing'])):
        ret = local_funcs.wheel(load)
    assert mock_ret == ret

def test_wheel_user_not_authenticated(local_funcs):
    if False:
        i = 10
        return i + 15
    "\n    Asserts that an UserAuthenticationError is returned when the user can't authenticate.\n    "
    mock_ret = {'error': {'name': 'UserAuthenticationError', 'message': 'Authentication failure of type "user" occurred for user UNKNOWN.'}}
    ret = local_funcs.wheel({})
    assert mock_ret == ret

def test_publish_user_is_blacklisted(local_funcs):
    if False:
        while True:
            i = 10
    '\n    Asserts that an AuthorizationError is returned when the user has been blacklisted.\n    '
    mock_ret = {'error': {'name': 'AuthorizationError', 'message': 'Authorization error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=True)):
        assert mock_ret == local_funcs.publish({'user': 'foo', 'fun': 'test.arg'})

def test_publish_cmd_blacklisted(local_funcs):
    if False:
        print('Hello World!')
    '\n    Asserts that an AuthorizationError is returned when the command has been blacklisted.\n    '
    mock_ret = {'error': {'name': 'AuthorizationError', 'message': 'Authorization error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=True)):
        assert mock_ret == local_funcs.publish({'user': 'foo', 'fun': 'test.arg'})

def test_publish_token_not_authenticated(local_funcs):
    if False:
        i = 10
        return i + 15
    "\n    Asserts that an AuthenticationError is returned when the token can't authenticate.\n    "
    load = {'user': 'foo', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'token': 'asdfasdfasdfasdf'}}
    mock_ret = {'error': {'name': 'AuthenticationError', 'message': 'Authentication error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)):
        assert mock_ret == local_funcs.publish(load)

def test_publish_token_authorization_error(local_funcs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts that an AuthorizationError is returned when the token authenticates, but is not\n    authorized.\n    '
    token = 'asdfasdfasdfasdf'
    load = {'user': 'foo', 'fun': 'test.arg', 'tgt': 'test_minion', 'arg': 'bar', 'kwargs': {'token': token}}
    mock_token = {'token': token, 'eauth': 'foo', 'name': 'test'}
    mock_ret = {'error': {'name': 'AuthorizationError', 'message': 'Authorization error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)), patch('salt.auth.LoadAuth.authenticate_token', MagicMock(return_value=mock_token)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=[])):
        assert mock_ret == local_funcs.publish(load)

def test_publish_eauth_not_authenticated(local_funcs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Asserts that an AuthenticationError is returned when the user can't authenticate.\n    "
    load = {'user': 'test', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'eauth': 'foo'}}
    mock_ret = {'error': {'name': 'AuthenticationError', 'message': 'Authentication error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)):
        assert mock_ret == local_funcs.publish(load)

def test_publish_eauth_authorization_error(local_funcs):
    if False:
        return 10
    '\n    Asserts that an AuthorizationError is returned when the user authenticates, but is not\n    authorized.\n    '
    load = {'user': 'test', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'eauth': 'foo'}, 'arg': 'bar'}
    mock_ret = {'error': {'name': 'AuthorizationError', 'message': 'Authorization error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)), patch('salt.auth.LoadAuth.authenticate_eauth', MagicMock(return_value=True)), patch('salt.auth.LoadAuth.get_auth_list', MagicMock(return_value=[])):
        assert mock_ret == local_funcs.publish(load)

def test_publish_user_not_authenticated(local_funcs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Asserts that an AuthenticationError is returned when the user can't authenticate.\n    "
    load = {'user': 'test', 'fun': 'test.arg', 'tgt': 'test_minion'}
    mock_ret = {'error': {'name': 'AuthenticationError', 'message': 'Authentication error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)):
        assert mock_ret == local_funcs.publish(load)

def test_publish_user_authenticated_missing_auth_list(local_funcs):
    if False:
        while True:
            i = 10
    '\n    Asserts that an AuthenticationError is returned when the user has an effective user id and is\n    authenticated, but the auth_list is empty.\n    '
    load = {'user': 'test', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'user': 'test'}, 'arg': 'foo'}
    mock_ret = {'error': {'name': 'AuthenticationError', 'message': 'Authentication error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)), patch('salt.auth.LoadAuth.authenticate_key', MagicMock(return_value='fake-user-key')), patch('salt.utils.master.get_values_of_matching_keys', MagicMock(return_value=[])):
        assert mock_ret == local_funcs.publish(load)

def test_publish_user_authorization_error(local_funcs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts that an AuthorizationError is returned when the user authenticates, but is not\n    authorized.\n    '
    load = {'user': 'test', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'user': 'test'}, 'arg': 'foo'}
    mock_ret = {'error': {'name': 'AuthorizationError', 'message': 'Authorization error occurred.'}}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)), patch('salt.auth.LoadAuth.authenticate_key', MagicMock(return_value='fake-user-key')), patch('salt.utils.master.get_values_of_matching_keys', MagicMock(return_value=['test'])), patch('salt.utils.minions.CkMinions.auth_check', MagicMock(return_value=False)):
        assert mock_ret == local_funcs.publish(load)

def test_dual_key_auth(check_local_funcs):
    if False:
        while True:
            i = 10
    '\n    Test for check for presented dual keys (salt, root) are authenticated\n    '
    load = {'user': 'test', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'user': 'test'}, 'arg': 'foo', 'key': 'mGXdurU1c8lXt5cmpbGq4rWvrOvDXxkwI9gbkP5CBBjpyGWuB8vkgz9r+sjjG0wVDL9/uFuREtk='}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)), patch('salt.utils.master.get_values_of_matching_keys', MagicMock(return_value=['test'])):
        results = check_local_funcs.publish(load)
        assert results == {'enc': 'clear', 'load': {'jid': None, 'minions': []}}

def test_dual_key_auth_sudo(check_local_funcs):
    if False:
        return 10
    '\n    Test for check for presented dual keys (salt, root) are authenticated\n    with a sudo user\n    '
    load = {'user': 'sudo_test', 'fun': 'test.arg', 'tgt': 'test_minion', 'kwargs': {'user': 'sudo_test'}, 'arg': 'foo', 'key': 'mGXdurU1c8lXt5cmpbGq4rWvrOvDXxkwI9gbkP5CBBjpyGWuB8vkgz9r+sjjG0wVDL9/uFuREtk='}
    with patch('salt.acl.PublisherACL.user_is_blacklisted', MagicMock(return_value=False)), patch('salt.acl.PublisherACL.cmd_is_blacklisted', MagicMock(return_value=False)), patch('salt.utils.master.get_values_of_matching_keys', MagicMock(return_value=['test'])):
        results = check_local_funcs.publish(load)
        assert results == {'enc': 'clear', 'load': {'jid': None, 'minions': []}}