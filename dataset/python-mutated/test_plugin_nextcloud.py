from unittest import mock
from apprise import Apprise
from apprise import NotifyType
import requests
from apprise.plugins.NotifyNextcloud import NotifyNextcloud
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('ncloud://:@/', {'instance': None}), ('ncloud://', {'instance': None}), ('nclouds://', {'instance': None}), ('ncloud://localhost', {'instance': NotifyNextcloud, 'notify_response': False}), ('ncloud://user@localhost?to=user1,user2&version=invalid', {'instance': TypeError}), ('ncloud://user@localhost?to=user1,user2&version=0', {'instance': TypeError}), ('ncloud://user@localhost?to=user1,user2&version=-23', {'instance': TypeError}), ('ncloud://localhost/admin', {'instance': NotifyNextcloud}), ('ncloud://user@localhost/admin', {'instance': NotifyNextcloud}), ('ncloud://user@localhost?to=user1,user2', {'instance': NotifyNextcloud}), ('ncloud://user@localhost?to=user1,user2&version=20', {'instance': NotifyNextcloud}), ('ncloud://user@localhost?to=user1,user2&version=21', {'instance': NotifyNextcloud}), ('ncloud://user@localhost?to=user1&version=20&url_prefix=/abcd', {'instance': NotifyNextcloud}), ('ncloud://user@localhost?to=user1&version=21&url_prefix=/abcd', {'instance': NotifyNextcloud}), ('ncloud://user:pass@localhost/user1/user2', {'instance': NotifyNextcloud, 'privacy_url': 'ncloud://user:****@localhost/user1/user2'}), ('ncloud://user:pass@localhost:8080/admin', {'instance': NotifyNextcloud}), ('nclouds://user:pass@localhost/admin', {'instance': NotifyNextcloud, 'privacy_url': 'nclouds://user:****@localhost/admin'}), ('nclouds://user:pass@localhost:8080/admin/', {'instance': NotifyNextcloud}), ('ncloud://localhost:8080/admin?+HeaderKey=HeaderValue', {'instance': NotifyNextcloud}), ('ncloud://user:pass@localhost:8081/admin', {'instance': NotifyNextcloud, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('ncloud://user:pass@localhost:8082/admin', {'instance': NotifyNextcloud, 'response': False, 'requests_response_code': 999}), ('ncloud://user:pass@localhost:8083/user1/user2/user3', {'instance': NotifyNextcloud, 'test_requests_exceptions': True}))

def test_plugin_nextcloud_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyNextcloud() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_nextcloud_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyNextcloud() Edge Cases\n\n    '
    robj = mock.Mock()
    robj.content = ''
    robj.status_code = requests.codes.ok
    mock_post.return_value = robj
    obj = NotifyNextcloud(host='localhost', user='admin', password='pass', targets='user')
    assert isinstance(obj, NotifyNextcloud) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='') is True
    assert 'data' in mock_post.call_args_list[0][1]
    assert 'shortMessage' in mock_post.call_args_list[0][1]['data']
    assert 'longMessage' not in mock_post.call_args_list[0][1]['data']

@mock.patch('requests.post')
def test_plugin_nextcloud_url_prefix(mock_post):
    if False:
        return 10
    '\n    NotifyNextcloud() URL Prefix Testing\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('ncloud://localhost/admin/?version=20&url_prefix=/abcd')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost/abcd/ocs/v2.php/apps/admin_notifications/api/v1/notifications/admin'
    mock_post.reset_mock()
    obj = Apprise.instantiate('ncloud://localhost/admin/?version=21&url_prefix=a/longer/path/abcd/')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost/a/longer/path/abcd/ocs/v2.php/apps/notifications/api/v2/admin_notifications/admin'