from unittest import mock
from json import dumps
from apprise import Apprise
import requests
from apprise.plugins.NotifyEmby import NotifyEmby
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('emby://', {'instance': None}), ('embys://', {'instance': None}), ('emby://localhost', {'instance': TypeError}), ('emby://:@/', {'instance': None}), ('emby://l2g@localhost', {'instance': NotifyEmby, 'response': False}), ('embys://l2g:password@localhost', {'instance': NotifyEmby, 'response': False, 'privacy_url': 'embys://l2g:****@localhost'}))

def test_plugin_template_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTemplate() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('apprise.plugins.NotifyEmby.NotifyEmby.sessions')
@mock.patch('apprise.plugins.NotifyEmby.NotifyEmby.login')
@mock.patch('apprise.plugins.NotifyEmby.NotifyEmby.logout')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_emby_general(mock_post, mock_get, mock_logout, mock_login, mock_sessions):
    if False:
        i = 10
        return i + 15
    '\n    NotifyEmby General Tests\n\n    '
    req = requests.Request()
    req.status_code = requests.codes.ok
    req.content = ''
    mock_get.return_value = req
    mock_post.return_value = req
    mock_login.return_value = True
    mock_logout.return_value = True
    mock_sessions.return_value = {'abcd': {}}
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost?modal=False')
    assert isinstance(obj, NotifyEmby)
    assert obj.notify('title', 'body', 'info') is True
    obj.access_token = 'abc'
    obj.user_id = '123'
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost?modal=True')
    assert isinstance(obj, NotifyEmby)
    assert obj.notify('title', 'body', 'info') is True
    obj.access_token = 'abc'
    obj.user_id = '123'
    for _exception in AppriseURLTester.req_exceptions:
        mock_post.side_effect = _exception
        mock_get.side_effect = _exception
        assert obj.notify('title', 'body', 'info') is False
    mock_post.side_effect = None
    mock_get.side_effect = None
    mock_post.return_value.content = u''
    mock_get.return_value.content = mock_post.return_value.content
    mock_post.return_value.status_code = 999
    mock_get.return_value.status_code = 999
    assert obj.notify('title', 'body', 'info') is False
    mock_post.return_value.status_code = requests.codes.internal_server_error
    mock_get.return_value.status_code = requests.codes.internal_server_error
    assert obj.notify('title', 'body', 'info') is False
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value.content = mock_post.return_value.content
    obj.port = None
    assert obj.notify('title', 'body', 'info') is True
    mock_sessions.return_value = {}
    assert obj.notify('title', 'body', 'info') is True
    del obj

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_emby_login(mock_post, mock_get):
    if False:
        i = 10
        return i + 15
    '\n    NotifyEmby() login()\n\n    '
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost')
    assert isinstance(obj, NotifyEmby)
    for _exception in AppriseURLTester.req_exceptions:
        mock_post.side_effect = _exception
        mock_get.side_effect = _exception
        assert obj.login() is False
    mock_post.side_effect = None
    mock_get.side_effect = None
    mock_post.return_value.content = u''
    mock_get.return_value.content = mock_post.return_value.content
    mock_post.return_value.status_code = 999
    mock_get.return_value.status_code = 999
    assert obj.login() is False
    mock_post.return_value.status_code = requests.codes.internal_server_error
    mock_get.return_value.status_code = requests.codes.internal_server_error
    assert obj.login() is False
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost:1234')
    assert isinstance(obj, NotifyEmby)
    assert obj.port == 1234
    assert obj.login() is False
    obj.port = None
    assert obj.login() is False
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost')
    assert isinstance(obj, NotifyEmby)
    assert obj.port == 8096
    assert obj.login() is False
    mock_post.return_value.content = dumps({u'AccessToken': u'0000-0000-0000-0000'})
    mock_get.return_value.content = mock_post.return_value.content
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost')
    assert isinstance(obj, NotifyEmby)
    assert obj.login() is False
    mock_post.return_value.content = dumps({u'User': {u'Id': u'abcd123'}, u'Id': u'123abc', u'AccessToken': u'0000-0000-0000-0000'})
    mock_get.return_value.content = mock_post.return_value.content
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost')
    assert isinstance(obj, NotifyEmby)
    assert obj.login() is True
    assert obj.user_id == '123abc'
    assert obj.access_token == '0000-0000-0000-0000'
    mock_post.return_value.content = dumps({u'User': {u'Id': u'abcd123'}, u'AccessToken': u'0000-0000-0000-0000'})
    mock_get.return_value.content = mock_post.return_value.content
    assert obj.login() is True
    assert obj.user_id == 'abcd123'
    assert obj.access_token == '0000-0000-0000-0000'

@mock.patch('apprise.plugins.NotifyEmby.NotifyEmby.login')
@mock.patch('apprise.plugins.NotifyEmby.NotifyEmby.logout')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_emby_sessions(mock_post, mock_get, mock_logout, mock_login):
    if False:
        i = 10
        return i + 15
    '\n    NotifyEmby() sessions()\n\n    '
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_login.return_value = True
    mock_logout.return_value = True
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost')
    assert isinstance(obj, NotifyEmby)
    obj.access_token = 'abc'
    obj.user_id = '123'
    for _exception in AppriseURLTester.req_exceptions:
        mock_post.side_effect = _exception
        mock_get.side_effect = _exception
        sessions = obj.sessions()
        assert isinstance(sessions, dict) is True
        assert len(sessions) == 0
    mock_post.side_effect = None
    mock_get.side_effect = None
    mock_post.return_value.content = u''
    mock_get.return_value.content = mock_post.return_value.content
    mock_post.return_value.status_code = 999
    mock_get.return_value.status_code = 999
    sessions = obj.sessions()
    assert isinstance(sessions, dict) is True
    assert len(sessions) == 0
    mock_post.return_value.status_code = requests.codes.internal_server_error
    mock_get.return_value.status_code = requests.codes.internal_server_error
    sessions = obj.sessions()
    assert isinstance(sessions, dict) is True
    assert len(sessions) == 0
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value.content = mock_post.return_value.content
    obj.port = None
    sessions = obj.sessions()
    assert isinstance(sessions, dict) is True
    assert len(sessions) == 0
    mock_post.return_value.content = dumps([{u'Id': u'abc123'}, {u'Id': u'def456'}, {u'InvalidEntry': None}])
    mock_get.return_value.content = mock_post.return_value.content
    sessions = obj.sessions(user_controlled=True)
    assert isinstance(sessions, dict) is True
    assert len(sessions) == 2
    sessions = obj.sessions(user_controlled=False)
    assert isinstance(sessions, dict) is True
    assert len(sessions) == 2
    obj.user_id = None
    mock_login.return_value = False
    sessions = obj.sessions()
    assert isinstance(sessions, dict) is True
    assert len(sessions) == 0

@mock.patch('apprise.plugins.NotifyEmby.NotifyEmby.login')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_emby_logout(mock_post, mock_get, mock_login):
    if False:
        while True:
            i = 10
    '\n    NotifyEmby() logout()\n\n    '
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_login.return_value = True
    obj = Apprise.instantiate('emby://l2g:l2gpass@localhost')
    assert isinstance(obj, NotifyEmby)
    obj.access_token = 'abc'
    obj.user_id = '123'
    for _exception in AppriseURLTester.req_exceptions:
        mock_post.side_effect = _exception
        mock_get.side_effect = _exception
        obj.logout()
        obj.access_token = 'abc'
        obj.user_id = '123'
    mock_post.side_effect = None
    mock_get.side_effect = None
    mock_post.return_value.content = u''
    mock_get.return_value.content = mock_post.return_value.content
    mock_post.return_value.status_code = 999
    mock_get.return_value.status_code = 999
    obj.logout()
    obj.access_token = 'abc'
    obj.user_id = '123'
    mock_post.return_value.status_code = requests.codes.internal_server_error
    mock_get.return_value.status_code = requests.codes.internal_server_error
    obj.logout()
    obj.access_token = 'abc'
    obj.user_id = '123'
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value.content = mock_post.return_value.content
    obj.port = None
    obj.logout()
    obj.logout()
    mock_post.side_effect = LookupError()
    mock_get.side_effect = LookupError()
    obj.access_token = 'abc'
    obj.user_id = '123'
    del obj