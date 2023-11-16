import os
from unittest import mock
import pytest
import requests
from json import dumps
from apprise import Apprise
from apprise import AppriseAttachment
from apprise.plugins.NotifyPushBullet import NotifyPushBullet
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('pbul://', {'instance': TypeError}), ('pbul://:@/', {'instance': TypeError}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'requests_response_text': {'file_name': 'cat.jpeg', 'file_type': 'image/jpeg', 'file_url': 'http://file_url', 'upload_url': 'http://upload_url'}}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'requests_response_text': {'file_name': 'test.pdf', 'file_type': 'application/pdf', 'file_url': 'http://file_url', 'upload_url': 'http://upload_url'}}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'requests_response_text': {'file_name': 'test.pdf', 'file_type': 'application/pdf', 'file_url': 'http://file_url'}, 'attach_response': False}), ('pbul://%s/#channel/' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s/?to=#channel' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s/#channel1/#channel2' % ('a' * 32), {'instance': NotifyPushBullet, 'privacy_url': 'pbul://a...a/', 'check_attachments': False}), ('pbul://%s/device/' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s/device1/device2/' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s/user@example.com/' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s/user@example.com/abc@def.com/' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s/device/#channel/user@example.com/' % ('a' * 32), {'instance': NotifyPushBullet, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'response': False, 'requests_response_code': requests.codes.internal_server_error, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'response': False, 'requests_response_code': 999, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'test_requests_exceptions': True, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'response': False, 'requests_response_code': requests.codes.internal_server_error, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'response': False, 'requests_response_code': 999, 'check_attachments': False}), ('pbul://%s' % ('a' * 32), {'instance': NotifyPushBullet, 'test_requests_exceptions': True, 'check_attachments': False}))

def test_plugin_pushbullet_urls():
    if False:
        print('Hello World!')
    '\n    NotifyPushBullet() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_pushbullet_attachments(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyPushBullet() Attachment Checks\n\n    '
    access_token = 't' * 32
    response = mock.Mock()
    response.content = dumps({'file_name': 'cat.jpg', 'file_type': 'image/jpeg', 'file_url': 'https://dl.pushb.com/abc/cat.jpg', 'upload_url': 'https://upload.pushbullet.com/abcd123'})
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    obj = Apprise.instantiate('pbul://{}/?format=markdown'.format(access_token))
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 4
    assert mock_post.call_args_list[0][0][0] == 'https://api.pushbullet.com/v2/upload-request'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.pushbullet.com/abcd123'
    assert mock_post.call_args_list[2][0][0] == 'https://api.pushbullet.com/v2/pushes'
    assert mock_post.call_args_list[3][0][0] == 'https://api.pushbullet.com/v2/pushes'
    mock_post.reset_mock()
    attach.add(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 7
    assert mock_post.call_args_list[0][0][0] == 'https://api.pushbullet.com/v2/upload-request'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.pushbullet.com/abcd123'
    assert mock_post.call_args_list[2][0][0] == 'https://api.pushbullet.com/v2/upload-request'
    assert mock_post.call_args_list[3][0][0] == 'https://upload.pushbullet.com/abcd123'
    assert mock_post.call_args_list[4][0][0] == 'https://api.pushbullet.com/v2/pushes'
    assert mock_post.call_args_list[5][0][0] == 'https://api.pushbullet.com/v2/pushes'
    assert mock_post.call_args_list[6][0][0] == 'https://api.pushbullet.com/v2/pushes'
    mock_post.reset_mock()
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    attach = AppriseAttachment(path)
    assert obj.notify(body='test', attach=attach) is False
    assert mock_post.call_count == 0
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    bad_response = mock.Mock()
    bad_response.content = dumps({'file_name': 'cat.jpg', 'file_type': 'image/jpeg', 'file_url': 'https://dl.pushb.com/abc/cat.jpg', 'upload_url': 'https://upload.pushbullet.com/abcd123'})
    bad_response.status_code = requests.codes.internal_server_error
    bad_json_response = mock.Mock()
    bad_json_response.content = '}'
    bad_json_response.status_code = requests.codes.ok
    mock_post.return_value = None
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = side_effect
        assert obj.send(body='test', attach=attach) is False
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [response, side_effect]
        assert obj.send(body='test', attach=attach) is False
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [response, response, side_effect]
        assert obj.send(body='test', attach=attach) is False
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [response, response, response, side_effect]
        assert obj.send(body='test', attach=attach) is False
    mock_post.side_effect = bad_json_response
    assert obj.send(body='test', attach=attach) is False

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_pushbullet_edge_cases(mock_post, mock_get):
    if False:
        return 10
    '\n    NotifyPushBullet() Edge Cases\n\n    '
    accesstoken = 'a' * 32
    recipients = '#chan1,#chan2,device,user@example.com,,,'
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifyPushBullet(accesstoken=None)
    with pytest.raises(TypeError):
        NotifyPushBullet(accesstoken='     ')
    obj = NotifyPushBullet(accesstoken=accesstoken, targets=recipients)
    assert isinstance(obj, NotifyPushBullet) is True
    assert len(obj.targets) == 4
    obj = NotifyPushBullet(accesstoken=accesstoken)
    assert isinstance(obj, NotifyPushBullet) is True
    assert len(obj.targets) == 1
    obj = NotifyPushBullet(accesstoken=accesstoken, targets=set())
    assert isinstance(obj, NotifyPushBullet) is True
    assert len(obj.targets) == 1