import pytest
import requests
from apprise import NotifyType
from apprise.plugins.NotifyRocketChat import NotifyRocketChat
from helpers import AppriseURLTester
from unittest import mock
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('rocket://', {'instance': None}), ('rockets://', {'instance': None}), ('rocket://:@/', {'instance': None}), ('rocket://localhost', {'instance': TypeError}), ('rocket://user:pass@localhost', {'instance': TypeError}), ('rocket://user:pass@localhost/#/!/@', {'instance': TypeError}), ('rocket://user@localhost/room/', {'instance': TypeError}), ('rocket://localhost/room/', {'instance': TypeError}), ('rocket://user:pass@localhost:8080/room/', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rocket://user:****@localhost'}), ('rockets://user:pass@localhost?to=#channel', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rockets://user:****@localhost'}), ('rockets://user:pass@localhost/#channel', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rockets://user:****@localhost'}), ('rocket://user:pass@localhost/#channel1/#channel2/?avatar=Yes', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rocket://user:****@localhost'}), ('rocket://user:pass@localhost/room1/room2', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rocket://user:****@localhost'}), ('rocket://user:pass@localhost/room/#channel?mode=basic&avatar=Yes', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rocket://user:****@localhost'}), ('rockets://user:pass%2Fwithslash@localhost/#channel/?mode=basic', {'instance': NotifyRocketChat, 'requests_response_text': {'status': 'success', 'data': {'authToken': 'abcd', 'userId': 'user'}}, 'privacy_url': 'rockets://user:****@localhost'}), ('rockets://user:pass@localhost/rooma/#channela', {'requests_response_code': requests.codes.ok, 'requests_response_text': {'status': 'failure'}, 'instance': NotifyRocketChat, 'response': False}), ('rockets://web/token@localhost/@user/#channel/roomid', {'instance': NotifyRocketChat, 'privacy_url': 'rockets://****@localhost/#channel/roomid'}), ('rockets://user:web/token@localhost/@user/?mode=webhook', {'instance': NotifyRocketChat, 'privacy_url': 'rockets://user:****@localhost'}), ('rockets://user:web/token@localhost?to=@user2,#channel2', {'instance': NotifyRocketChat}), ('rockets://web/token@localhost/?avatar=No', {'instance': NotifyRocketChat, 'privacy_url': 'rockets://****@localhost/'}), ('rockets://localhost/@user/?mode=webhook&webhook=web/token', {'instance': NotifyRocketChat, 'privacy_url': 'rockets://****@localhost/@user'}), ('rockets://user:web/token@localhost/@user/?mode=invalid', {'instance': TypeError}), ('rocket://user:pass@localhost:8081/room1/room2', {'instance': NotifyRocketChat, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('rockets://user:web/token@localhost?to=@user3,#channel3', {'instance': NotifyRocketChat, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('rocket://user:pass@localhost:8082/#channel', {'instance': NotifyRocketChat, 'response': False, 'requests_response_code': 999}), ('rocket://user:pass@localhost:8083/#chan1/#chan2/room', {'instance': NotifyRocketChat, 'test_requests_exceptions': True}))

def test_plugin_rocket_chat_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyRocketChat() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_rocket_chat_edge_cases(mock_post, mock_get):
    if False:
        return 10
    '\n    NotifyRocketChat() Edge Cases\n\n    '
    recipients = 'AbcD1245, @l2g, @lead2gold, #channel, #channel2'
    user = 'myuser'
    password = 'mypass'
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = ''
    mock_get.return_value.content = ''
    obj = NotifyRocketChat(user=user, password=password, targets=recipients)
    assert isinstance(obj, NotifyRocketChat) is True
    assert len(obj.channels) == 2
    assert len(obj.users) == 2
    assert len(obj.rooms) == 1
    with pytest.raises(TypeError):
        obj = NotifyRocketChat(webhook=None, mode='webhook')
    assert obj.logout() is True
    mock_post.return_value.content = '{'
    mock_get.return_value.content = '}'
    assert obj.login() is False
    mock_post.return_value.content = ''
    mock_get.return_value.content = ''
    mock_post.return_value.status_code = requests.codes.internal_server_error
    mock_get.return_value.status_code = requests.codes.internal_server_error
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert obj._send(payload='test', notify_type=NotifyType.INFO) is False
    assert obj.logout() is False
    mock_post.return_value.status_code = 999
    mock_get.return_value.status_code = 999
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert obj._send(payload='test', notify_type=NotifyType.INFO) is False
    assert obj.logout() is False
    mock_get.side_effect = requests.ConnectionError(0, 'requests.ConnectionError() not handled')
    mock_post.side_effect = mock_get.side_effect
    assert obj._send(payload='test', notify_type=NotifyType.INFO) is False
    obj.login = mock.Mock()
    obj.login.return_value = True
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert obj.logout() is False