import os
import json
from unittest import mock
import requests
import apprise
from helpers import AppriseURLTester
from apprise.plugins.NotifyNtfy import NtfyPriority, NotifyNtfy
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
GOOD_RESPONSE_TEXT = {'code': '0', 'error': 'success'}
apprise_url_tests = (('ntfy://', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'response': False}), ('ntfys://', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'response': False}), ('ntfy://:@/', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'response': False}), ('ntfy://user:pass@localhost?mode=private', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'response': False}), ('ntfy://user:pass@localhost/#/!/@', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'response': False}), ('ntfy://user@localhost/topic/', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://user@localhost/topic'}), ('ntfy://ntfy.sh/topic1/topic2/', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/topic2/', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?email=user@gmail.com', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?tags=tag1,tag2,tag3', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?delay=3600', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?title=A%20Great%20Title', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?click=yes', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?email=user@example.com', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?image=False', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?avatar_url=ttp://localhost/test.jpg', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?attach=http://example.com/file.jpg', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?attach=http://example.com/file.jpg&filename=smoke.jpg', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://localhost/topic1/?attach=http://-%20', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://tk_abcd123456@localhost/topic1', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://t...6@localhost/topic1'}), ('ntfy://abcd123456@localhost/topic1?auth=token', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://a...6@localhost/topic1'}), ('ntfy://:abcd123456@localhost/topic1?auth=token', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://a...6@localhost/topic1'}), ('ntfy://localhost/topic1?token=abc1234', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://a...4@localhost/topic1'}), ('ntfy://user:token@localhost/topic1?auth=token', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://t...n@localhost/topic1'}), ('ntfy://localhost/topic1?auth=token', {'instance': NotifyNtfy, 'response': False, 'privacy_url': 'ntfy://localhost/topic1'}), ('ntfy://localhost/topic1/?priority=default', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT, 'privacy_url': 'ntfy://localhost/topic1'}), ('ntfy://localhost/topic1/?priority=high', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://user:pass@localhost:8080/topic/', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfys://user:pass@localhost?to=topic', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('https://just/a/random/host/that/means/nothing', {'instance': None}), ('https://ntfy.sh?to=topic', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://user:pass@topic1/topic2/topic3/?mode=cloud', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://user:pass@ntfy.sh/topic1/topic2/?mode=cloud', {'instance': NotifyNtfy, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfys://user:web/token@localhost/topic/?mode=invalid', {'instance': TypeError}), ('ntfys://token@localhost/topic/?auth=invalid', {'instance': TypeError}), ('ntfys://user:web@-_/topic1/topic2/?mode=private', {'instance': None}), ('ntfy://user:pass@localhost:8089/topic/topic2', {'instance': NotifyNtfy, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('ntfy://user:pass@localhost:8082/topic', {'instance': NotifyNtfy, 'response': False, 'requests_response_code': 999, 'requests_response_text': GOOD_RESPONSE_TEXT}), ('ntfy://user:pass@localhost:8083/topic1/topic2/', {'instance': NotifyNtfy, 'test_requests_exceptions': True, 'requests_response_text': GOOD_RESPONSE_TEXT}))

def test_plugin_ntfy_chat_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyNtfy() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_ntfy_attachments(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyNtfy() Attachment Checks\n\n    '
    response = mock.Mock()
    response.content = GOOD_RESPONSE_TEXT
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    mock_post.reset_mock()
    obj = apprise.Apprise.instantiate('ntfy://user:pass@localhost:8080/topic')
    assert obj.notify(title='hello', body='world')
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost:8080'
    response = json.loads(mock_post.call_args_list[0][1]['data'])
    assert response['topic'] == 'topic'
    assert response['title'] == 'hello'
    assert response['message'] == 'world'
    assert 'attach' not in response
    mock_post.reset_mock()
    attach = apprise.AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    obj = apprise.Apprise.instantiate('ntfy://user:pass@localhost:8084/topic')
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost:8084/topic'
    assert mock_post.call_args_list[0][1]['params']['message'] == 'test'
    assert 'title' not in mock_post.call_args_list[0][1]['params']
    assert mock_post.call_args_list[0][1]['params']['filename'] == 'apprise-test.gif'
    mock_post.reset_mock()
    attach.add(os.path.join(TEST_VAR_DIR, 'apprise-test.png'))
    assert obj.notify(body='test', title='wonderful', attach=attach) is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'http://localhost:8084/topic'
    assert mock_post.call_args_list[0][1]['params']['message'] == 'test'
    assert mock_post.call_args_list[0][1]['params']['title'] == 'wonderful'
    assert mock_post.call_args_list[0][1]['params']['filename'] == 'apprise-test.gif'
    assert mock_post.call_args_list[1][0][0] == 'http://localhost:8084/topic'
    assert 'message' not in mock_post.call_args_list[1][1]['params']
    assert 'title' not in mock_post.call_args_list[1][1]['params']
    assert mock_post.call_args_list[1][1]['params']['filename'] == 'apprise-test.png'
    mock_post.reset_mock()
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    attach = apprise.AppriseAttachment(path)
    assert obj.notify(body='test', attach=attach) is False
    assert mock_post.call_count == 0
    attach = apprise.AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    mock_post.return_value = None
    for side_effect in (requests.RequestException(), OSError()):
        mock_post.side_effect = side_effect
        assert obj.send(body='test', attach=attach) is False

@mock.patch('requests.post')
def test_plugin_custom_ntfy_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyNtfy() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    response.content = json.dumps(GOOD_RESPONSE_TEXT)
    mock_post.return_value = response
    results = NotifyNtfy.parse_url('ntfys://abc---,topic2,~~,,?priority=max&tags=smile,de')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] is None
    assert results['host'] == 'abc---,topic2,~~,,'
    assert results['fullpath'] is None
    assert results['path'] is None
    assert results['query'] is None
    assert results['schema'] == 'ntfys'
    assert results['url'] == 'ntfys://abc---,topic2,~~,,'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd']['priority'] == 'max'
    assert results['qsd']['tags'] == 'smile,de'
    instance = NotifyNtfy(**results)
    assert isinstance(instance, NotifyNtfy)
    assert len(instance.topics) == 2
    assert 'abc---' in instance.topics
    assert 'topic2' in instance.topics
    results = NotifyNtfy.parse_url('ntfy://localhost/topic1/?attach=http://example.com/file.jpg&filename=smoke.jpg')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] is None
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/topic1/'
    assert results['path'] == '/topic1/'
    assert results['query'] is None
    assert results['schema'] == 'ntfy'
    assert results['url'] == 'ntfy://localhost/topic1/'
    assert results['attach'] == 'http://example.com/file.jpg'
    assert results['filename'] == 'smoke.jpg'
    instance = NotifyNtfy(**results)
    assert isinstance(instance, NotifyNtfy)
    assert len(instance.topics) == 1
    assert 'topic1' in instance.topics
    assert instance.notify(body='body', title='title', notify_type=apprise.NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost'
    response = json.loads(mock_post.call_args_list[0][1]['data'])
    assert response['topic'] == 'topic1'
    assert response['message'] == 'body'
    assert response['title'] == 'title'
    assert response['attach'] == 'http://example.com/file.jpg'
    assert response['filename'] == 'smoke.jpg'

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_ntfy_config_files(mock_post, mock_get):
    if False:
        while True:
            i = 10
    '\n    NotifyNtfy() Config File Cases\n    '
    content = '\n    urls:\n      - ntfy://localhost/topic1:\n          - priority: 1\n            tag: ntfy_int min\n          - priority: "1"\n            tag: ntfy_str_int min\n          - priority: min\n            tag: ntfy_str min\n\n          # This will take on normal (default) priority\n          - priority: invalid\n            tag: ntfy_invalid\n\n      - ntfy://localhost/topic2:\n          - priority: 5\n            tag: ntfy_int max\n          - priority: "5"\n            tag: ntfy_str_int max\n          - priority: emergency\n            tag: ntfy_str max\n          - priority: max\n            tag: ntfy_str max\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value = requests.Request()
    mock_get.return_value.status_code = requests.codes.ok
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 8
    assert len(aobj) == 8
    assert len([x for x in aobj.find(tag='min')]) == 3
    for s in aobj.find(tag='min'):
        assert s.priority == NtfyPriority.MIN
    assert len([x for x in aobj.find(tag='max')]) == 4
    for s in aobj.find(tag='max'):
        assert s.priority == NtfyPriority.MAX
    assert len([x for x in aobj.find(tag='ntfy_str')]) == 3
    assert len([x for x in aobj.find(tag='ntfy_str_int')]) == 2
    assert len([x for x in aobj.find(tag='ntfy_int')]) == 2
    assert len([x for x in aobj.find(tag='ntfy_invalid')]) == 1
    assert next(aobj.find(tag='ntfy_invalid')).priority == NtfyPriority.NORMAL