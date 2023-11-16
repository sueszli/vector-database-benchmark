import os
from unittest import mock
import requests
import pytest
from json import dumps
from apprise.plugins.NotifyPushover import PushoverPriority, NotifyPushover
import apprise
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('pover://', {'instance': TypeError}), ('pover://:@/', {'instance': TypeError}), ('pover://%s' % ('a' * 30), {'instance': TypeError}), ('pover://%s@%s?sound=mysound' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?sound=spacealarm' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?url=my-url&url_title=title' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover, 'include_image': False}), ('pover://%s@%s/DEVICE' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?to=DEVICE' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s/DEVICE1/Device-with-dash/' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover, 'privacy_url': 'pover://u...u@a...a'}), ('pover://%s@%s/%s/' % ('u' * 30, 'a' * 30, 'd' * 30), {'instance': NotifyPushover, 'response': False}), ('pover://%s@%s/DEVICE1/%s/' % ('u' * 30, 'a' * 30, 'd' * 30), {'instance': NotifyPushover, 'response': False}), ('pover://%s@%s?priority=high' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?priority=high&format=html' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?priority=high&format=markdown' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?priority=invalid' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?priority=emergency' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?priority=2' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s?priority=emergency&%s&%s' % ('u' * 30, 'a' * 30, 'retry=30', 'expire=300'), {'instance': NotifyPushover}), ('pover://%s@%s?priority=emergency&%s&%s' % ('u' * 30, 'a' * 30, 'retry=invalid', 'expire=300'), {'instance': NotifyPushover}), ('pover://%s@%s?priority=emergency&%s&%s' % ('u' * 30, 'a' * 30, 'retry=30', 'expire=invalid'), {'instance': NotifyPushover}), ('pover://%s@%s?priority=emergency&%s' % ('u' * 30, 'a' * 30, 'expire=100000'), {'instance': TypeError}), ('pover://%s@%s?priority=emergency&%s' % ('u' * 30, 'a' * 30, 'retry=15'), {'instance': TypeError}), ('pover://%s@%s?priority=' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover}), ('pover://%s@%s' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pover://%s@%s' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover, 'response': False, 'requests_response_code': 999}), ('pover://%s@%s' % ('u' * 30, 'a' * 30), {'instance': NotifyPushover, 'test_requests_exceptions': True}))

def test_plugin_pushover_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyPushover() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_pushover_attachments(mock_post, tmpdir):
    if False:
        print('Hello World!')
    '\n    NotifyPushover() Attachment Checks\n\n    '
    user_key = 'u' * 30
    api_token = 'a' * 30
    response = mock.Mock()
    response.content = dumps({'status': 1, 'request': '647d2300-702c-4b38-8b2f-d56326ae460b'})
    response.status_code = requests.codes.ok
    bad_response = mock.Mock()
    response.content = dumps({'status': 1, 'request': '647d2300-702c-4b38-8b2f-d56326ae460b'})
    bad_response.status_code = requests.codes.internal_server_error
    mock_post.return_value = response
    attach = apprise.AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    obj = apprise.Apprise.instantiate('pover://{}@{}/'.format(user_key, api_token))
    assert isinstance(obj, NotifyPushover)
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.pushover.net/1/messages.json'
    mock_post.reset_mock()
    assert attach.add(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.pushover.net/1/messages.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.pushover.net/1/messages.json'
    mock_post.reset_mock()
    image = tmpdir.mkdir('pover_image').join('test.jpg')
    image.write('a' * NotifyPushover.attach_max_size_bytes)
    attach = apprise.AppriseAttachment.instantiate(str(image))
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.pushover.net/1/messages.json'
    mock_post.reset_mock()
    image.write('a' * (NotifyPushover.attach_max_size_bytes + 1))
    attach = apprise.AppriseAttachment.instantiate(str(image))
    assert obj.notify(body='test', attach=attach) is False
    assert mock_post.call_count == 0
    attach = apprise.AppriseAttachment.instantiate('file://{}?cache=False'.format(str(image)))
    os.unlink(str(image))
    assert obj.notify(body='body', title='title', attach=attach) is False
    assert mock_post.call_count == 0
    image = tmpdir.mkdir('pover_unsupported').join('test.doc')
    image.write('a' * 256)
    attach = apprise.AppriseAttachment.instantiate(str(image))
    assert obj.notify(body='test', attach=attach) is True
    attach = apprise.AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [side_effect, side_effect]
        assert obj.send(body='test', attach=attach) is False
        assert obj.send(body='test') is False

@mock.patch('requests.post')
def test_plugin_pushover_edge_cases(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyPushover() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyPushover(token=None)
    token = 'a' * 30
    user_key = 'u' * 30
    invalid_device = 'd' * 35
    devices = 'device1,device2,,,,%s' % invalid_device
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifyPushover(user_key=user_key, webhook_id=None)
    obj = NotifyPushover(user_key=user_key, token=token, targets=devices)
    assert isinstance(obj, NotifyPushover) is True
    assert len(obj.targets) == 2
    assert obj.notify(body='body', title='title', notify_type=apprise.NotifyType.INFO) is True
    obj = NotifyPushover(user_key=user_key, token=token)
    assert isinstance(obj, NotifyPushover) is True
    assert len(obj.targets) == 1
    assert obj.notify(body='body', title='title', notify_type=apprise.NotifyType.INFO) is True
    obj = NotifyPushover(user_key=user_key, token=token, targets=set())
    assert isinstance(obj, NotifyPushover) is True
    assert len(obj.targets) == 1
    with pytest.raises(TypeError):
        NotifyPushover(user_key=None, token='abcd')
    with pytest.raises(TypeError):
        NotifyPushover(user_key='abcd', token=None)
    with pytest.raises(TypeError):
        NotifyPushover(user_key='abcd', token='  ')

@mock.patch('requests.post')
def test_plugin_pushover_config_files(mock_post):
    if False:
        return 10
    '\n    NotifyPushover() Config File Cases\n    '
    content = '\n    urls:\n      - pover://USER@TOKEN:\n          - priority: -2\n            tag: pushover_int low\n          - priority: "-2"\n            tag: pushover_str_int low\n          - priority: low\n            tag: pushover_str low\n\n          # This will take on normal (default) priority\n          - priority: invalid\n            tag: pushover_invalid\n\n      - pover://USER2@TOKEN2:\n          - priority: 2\n            tag: pushover_int emerg\n          - priority: "2"\n            tag: pushover_str_int emerg\n          - priority: emergency\n            tag: pushover_str emerg\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 7
    assert len(aobj) == 7
    assert len([x for x in aobj.find(tag='low')]) == 3
    for s in aobj.find(tag='low'):
        assert s.priority == PushoverPriority.LOW
    assert len([x for x in aobj.find(tag='emerg')]) == 3
    for s in aobj.find(tag='emerg'):
        assert s.priority == PushoverPriority.EMERGENCY
    assert len([x for x in aobj.find(tag='pushover_str')]) == 2
    assert len([x for x in aobj.find(tag='pushover_str_int')]) == 2
    assert len([x for x in aobj.find(tag='pushover_int')]) == 2
    assert len([x for x in aobj.find(tag='pushover_invalid')]) == 1
    assert next(aobj.find(tag='pushover_invalid')).priority == PushoverPriority.NORMAL
    assert aobj.notify(title='title', body='body', tag=[('pushover_str_int', 'low')]) is True
    assert aobj.notify(title='title', body='body') is True