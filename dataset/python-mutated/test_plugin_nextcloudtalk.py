from unittest import mock
from apprise import Apprise
from apprise import NotifyType
import requests
from apprise.plugins.NotifyNextcloudTalk import NotifyNextcloudTalk
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('nctalk://:@/', {'instance': None}), ('nctalk://', {'instance': None}), ('nctalks://', {'instance': None}), ('nctalk://localhost', {'instance': TypeError}), ('nctalk://localhost/roomid', {'instance': TypeError}), ('nctalk://user@localhost/roomid', {'instance': TypeError}), ('nctalk://user:pass@localhost', {'instance': NotifyNextcloudTalk, 'notify_response': False}), ('nctalk://user:pass@localhost/roomid1/roomid2', {'instance': NotifyNextcloudTalk, 'requests_response_code': requests.codes.created, 'privacy_url': 'nctalk://user:****@localhost/roomid1/roomid2'}), ('nctalk://user:pass@localhost:8080/roomid', {'instance': NotifyNextcloudTalk, 'requests_response_code': requests.codes.created}), ('nctalk://user:pass@localhost:8080/roomid?url_prefix=/prefix', {'instance': NotifyNextcloudTalk, 'requests_response_code': requests.codes.created}), ('nctalks://user:pass@localhost/roomid', {'instance': NotifyNextcloudTalk, 'requests_response_code': requests.codes.created, 'privacy_url': 'nctalks://user:****@localhost/roomid'}), ('nctalks://user:pass@localhost:8080/roomid/', {'instance': NotifyNextcloudTalk, 'requests_response_code': requests.codes.created}), ('nctalk://user:pass@localhost:8080/roomid?+HeaderKey=HeaderValue', {'instance': NotifyNextcloudTalk, 'requests_response_code': requests.codes.created}), ('nctalk://user:pass@localhost:8081/roomid', {'instance': NotifyNextcloudTalk, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('nctalk://user:pass@localhost:8082/roomid', {'instance': NotifyNextcloudTalk, 'response': False, 'requests_response_code': 999}), ('nctalk://user:pass@localhost:8083/roomid1/roomid2/roomid3', {'instance': NotifyNextcloudTalk, 'test_requests_exceptions': True}))

def test_plugin_nextcloudtalk_urls():
    if False:
        return 10
    '\n    NotifyNextcloudTalk() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_nextcloudtalk_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyNextcloudTalk() Edge Cases\n\n    '
    robj = mock.Mock()
    robj.content = ''
    robj.status_code = requests.codes.created
    mock_post.return_value = robj
    obj = NotifyNextcloudTalk(host='localhost', user='admin', password='pass', targets='roomid')
    assert isinstance(obj, NotifyNextcloudTalk) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='') is True
    assert 'data' in mock_post.call_args_list[0][1]
    assert 'message' in mock_post.call_args_list[0][1]['data']

@mock.patch('requests.post')
def test_plugin_nextcloud_talk_url_prefix(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyNextcloudTalk() URL Prefix Testing\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.created
    mock_post.return_value = response
    obj = Apprise.instantiate('nctalk://user:pass@localhost/admin/?url_prefix=/abcd')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost/abcd/ocs/v2.php/apps/spreed/api/v1/chat/admin'
    mock_post.reset_mock()
    obj = Apprise.instantiate('nctalk://user:pass@localhost/admin/?url_prefix=a/longer/path/abcd/')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost/a/longer/path/abcd/ocs/v2.php/apps/spreed/api/v1/chat/admin'