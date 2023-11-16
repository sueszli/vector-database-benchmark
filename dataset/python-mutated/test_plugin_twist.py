from unittest import mock
import requests
from json import dumps
from apprise import Apprise
from apprise.plugins.NotifyTwist import NotifyTwist
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('twist://', {'instance': None}), ('twist://:@/', {'instance': None}), ('twist://user@example.com/', {'instance': None}), ('twist://user@example.com/password', {'instance': NotifyTwist, 'notify_response': False}), ('twist://password:user1@example.com', {'instance': NotifyTwist, 'notify_response': False, 'privacy_url': 'twist://****:user1@example.com'}), ('twist://password:user2@example.com', {'instance': NotifyTwist, 'requests_response_text': {'id': 1234, 'default_workspace': 9876}, 'notify_response': False}), ('twist://password:user2@example.com', {'instance': NotifyTwist, 'response': False, 'requests_response_code': 999}), ('twist://password:user2@example.com', {'instance': NotifyTwist, 'test_requests_exceptions': True}))

def test_plugin_twist_urls():
    if False:
        return 10
    '\n    NotifyTwist() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_twist_init():
    if False:
        print('Hello World!')
    '\n    NotifyTwist() init()\n\n    '
    try:
        NotifyTwist(email='invalid', targets=None)
        assert False
    except TypeError:
        assert True
    try:
        NotifyTwist(email='user@domain', targets=None)
        assert False
    except TypeError:
        assert True
    result = NotifyTwist(password='abc123', email='user@domain.com', targets=None)
    assert result.user == 'user'
    assert result.host == 'domain.com'
    assert result.password == 'abc123'
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel')
    assert isinstance(obj, NotifyTwist)
    obj = Apprise.instantiate('twist://password:user@example.com/12345')
    assert isinstance(obj, NotifyTwist)
    obj = Apprise.instantiate('twist://password:user@example.com/{}'.format('a' * 65))
    assert isinstance(obj, NotifyTwist)
    result = NotifyTwist.parse_url('twist://example.com')
    assert result is None
    result = NotifyTwist.parse_url('twist://password:user@example.com?to=#channel')
    assert isinstance(result, dict)
    assert 'user' in result
    assert result['user'] == 'user'
    assert 'host' in result
    assert result['host'] == 'example.com'
    assert 'password' in result
    assert result['password'] == 'password'
    assert 'targets' in result
    assert isinstance(result['targets'], list) is True
    assert len(result['targets']) == 1
    assert '#channel' in result['targets']

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_twist_auth(mock_post, mock_get):
    if False:
        print('Hello World!')
    '\n    NotifyTwist() login/logout()\n\n    '
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = dumps({'token': '2e82c1e4e8b0091fdaa34ff3972351821406f796', 'default_workspace': 12345})
    mock_get.return_value.content = mock_post.return_value.content
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel')
    assert isinstance(obj, NotifyTwist)
    obj.logout()
    assert obj.login() is True
    obj.channels.clear()
    assert obj._channel_migration() is True
    mock_post.return_value.content = dumps([{'name': 'TesT', 'id': 1}, {'name': 'tESt2', 'id': 2}])
    mock_get.return_value.content = mock_post.return_value.content
    results = obj.get_workspaces()
    assert len(results) == 2
    assert 'test' in results
    assert results['test'] == 1
    assert 'test2' in results
    assert results['test2'] == 2
    mock_post.return_value.content = dumps([{'name': 'ChaNNEL1', 'id': 1}, {'name': 'chaNNel2', 'id': 2}])
    mock_get.return_value.content = mock_post.return_value.content
    results = obj.get_channels(wid=1)
    assert len(results) == 2
    assert 'channel1' in results
    assert results['channel1'] == 1
    assert 'channel2' in results
    assert results['channel2'] == 2
    mock_post.return_value.status_code = 403
    mock_get.return_value.status_code = 403
    assert obj.get_workspaces() == dict()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    del obj
    mock_post.return_value.status_code = 403
    mock_get.return_value.status_code = 403
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel')
    assert isinstance(obj, NotifyTwist)
    assert obj.get_workspaces() == dict()
    assert obj.get_channels(wid=1) == dict()
    assert obj._channel_migration() is False
    assert obj.send('body', 'title') is False
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel')
    assert isinstance(obj, NotifyTwist)
    obj.logout()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_twist_cache(mock_post, mock_get):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTwist() Cache Handling\n\n    '

    def _response(url, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        request = mock.Mock()
        request.status_code = requests.codes.ok
        request.content = '{}'
        if url.endswith('/login'):
            request.content = dumps({'token': '2e82c1e4e8b0091fdaa34ff3972351821406f796', 'default_workspace': 1})
        elif url.endswith('workspaces/get'):
            request.content = dumps([{'name': 'TeamA', 'id': 1}, {'name': 'TeamB', 'id': 2}])
        elif url.endswith('channels/get'):
            request.content = dumps([{'name': 'ChanA', 'id': 1}, {'name': 'ChanB', 'id': 2}])
        return request
    mock_get.side_effect = _response
    mock_post.side_effect = _response
    obj = Apprise.instantiate('twist://password:user@example.com/#ChanB/1:1/TeamA:ChanA/Ignore:Chan/3:1')
    assert isinstance(obj, NotifyTwist)
    assert obj._channel_migration() is False
    obj.channels.add('ChanB')
    assert obj._channel_migration() is True
    assert obj._channel_migration() is True
    assert obj.send('body', 'title') is True

    def _can_not_send_response(url, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Simulate a case where we can't send a notification\n        "
        request = mock.Mock()
        request.status_code = 403
        request.content = '{}'
        return request
    mock_get.side_effect = _can_not_send_response
    mock_post.side_effect = _can_not_send_response
    assert obj.send('body', 'title') is False

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_twist_fetch(mock_post, mock_get):
    if False:
        print('Hello World!')
    '\n    NotifyTwist() fetch()\n\n    fetch() is a wrapper that handles all kinds of edge cases and even\n    attempts to re-authenticate to the Twist server if our token\n    happens to expire.  This tests these edge cases\n\n    '
    _cache = {'first_time': True}

    def _reauth_response(url, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Tests re-authentication process and then a successful\n        retry\n        '
        request = mock.Mock()
        request.status_code = requests.codes.ok
        request.content = dumps({'token': '2e82c1e4e8b0091fdaa34ff3972351821406f796', 'default_workspace': 12345})
        if url.endswith('threads/add') and _cache['first_time'] is True:
            _cache['first_time'] = False
            request.status_code = 403
            request.content = dumps({'error_code': 200, 'error_string': 'Invalid token'})
        return request
    mock_get.side_effect = _reauth_response
    mock_post.side_effect = _reauth_response
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel/34')
    assert isinstance(obj, NotifyTwist)
    (postokay, response) = obj._fetch('threads/add')
    _cache = {'first_time': True}

    def _reauth_exception_response(url, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Tests exception thrown after re-authentication process\n        '
        request = mock.Mock()
        request.status_code = requests.codes.ok
        request.content = dumps({'token': '2e82c1e4e8b0091fdaa34ff3972351821406f796', 'default_workspace': 12345})
        if url.endswith('threads/add') and _cache['first_time'] is True:
            _cache['first_time'] = False
            request.status_code = 403
            request.content = dumps({'error_code': 200, 'error_string': 'Invalid token'})
        elif url.endswith('threads/add') and _cache['first_time'] is False:
            request.status_code = 200
            request.content = '{'
        return request
    mock_get.side_effect = _reauth_exception_response
    mock_post.side_effect = _reauth_exception_response
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel/34')
    assert isinstance(obj, NotifyTwist)
    (postokay, response) = obj._fetch('threads/add')
    _cache = {'first_time': True}

    def _reauth_failed_response(url, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests re-authentication process and have it not succeed\n        '
        request = mock.Mock()
        request.status_code = requests.codes.ok
        request.content = dumps({'token': '2e82c1e4e8b0091fdaa34ff3972351821406f796', 'default_workspace': 12345})
        if url.endswith('threads/add') and _cache['first_time'] is True:
            _cache['first_time'] = False
            request.status_code = 403
            request.content = dumps({'error_code': 200, 'error_string': 'Invalid token'})
        elif url.endswith('/login') and _cache['first_time'] is False:
            request.status_code = 403
            request.content = '{}'
        return request
    mock_get.side_effect = _reauth_failed_response
    mock_post.side_effect = _reauth_failed_response
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel/34')
    assert isinstance(obj, NotifyTwist)
    (postokay, response) = obj._fetch('threads/add')

    def _unparseable_json_response(url, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        request = mock.Mock()
        request.status_code = requests.codes.ok
        request.content = '{'
        return request
    mock_get.side_effect = _unparseable_json_response
    mock_post.side_effect = _unparseable_json_response
    obj = Apprise.instantiate('twist://password:user@example.com/#Channel/34')
    assert isinstance(obj, NotifyTwist)
    (postokay, response) = obj._fetch('threads/add')
    assert postokay is True
    assert response == {}