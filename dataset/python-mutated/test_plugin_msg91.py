from unittest import mock
import pytest
import requests
from json import loads
from apprise import Apprise
from apprise.plugins.NotifyMSG91 import NotifyMSG91
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('msg91://', {'instance': TypeError}), ('msg91://-', {'instance': TypeError}), ('msg91://{}'.format('a' * 23), {'instance': TypeError}), ('msg91://{}@{}'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91, 'notify_response': False}), ('msg91://{}@{}/abcd'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91, 'notify_response': False}), ('msg91://{}@{}/15551232000'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91, 'privacy_url': 'msg91://t...t@a...a/15551232000'}), ('msg91://{}@{}/?to=15551232000&short_url=no'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91}), ('msg91://{}@{}/15551232000?short_url=yes'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91}), ('msg91://{}@{}/15551232000'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91}), ('msg91://{}@{}/15551232000'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91, 'response': False, 'requests_response_code': 999}), ('msg91://{}@{}/15551232000'.format('t' * 20, 'a' * 23), {'instance': NotifyMSG91, 'test_requests_exceptions': True}))

def test_plugin_msg91_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyMSG91() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_msg91_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyMSG91() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    target = '+1 (555) 123-3456'
    with pytest.raises(TypeError):
        NotifyMSG91(template='1234', authkey=None, targets=target)
    with pytest.raises(TypeError):
        NotifyMSG91(template='1234', authkey='    ', targets=target)
    with pytest.raises(TypeError):
        NotifyMSG91(template='     ', authkey='a' * 23, targets=target)
    with pytest.raises(TypeError):
        NotifyMSG91(template=None, authkey='a' * 23, targets=target)

@mock.patch('requests.post')
def test_plugin_msg91_keywords(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyMSG91() Templating\n\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    target = '+1 (555) 123-3456'
    template = '12345'
    authkey = '{}'.format('b' * 32)
    message_contents = 'test'
    obj = Apprise.instantiate('msg91://{}@{}/{}?:key=value&:mobiles=ignored'.format(template, authkey, target))
    assert isinstance(obj, NotifyMSG91) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body=message_contents) is True
    assert mock_post.call_count == 1
    first_call = mock_post.call_args_list[0]
    assert first_call[0][0] == 'https://control.msg91.com/api/v5/flow/'
    response = loads(first_call[1]['data'])
    assert response['template_id'] == template
    assert response['short_url'] == 0
    assert len(response['recipients']) == 1
    assert response['recipients'][0]['mobiles'] == '15551233456'
    assert response['recipients'][0]['body'] == message_contents
    assert response['recipients'][0]['type'] == 'info'
    assert response['recipients'][0]['key'] == 'value'
    mock_post.reset_mock()
    obj = Apprise.instantiate('msg91://{}@{}/{}?:body&:type=cat'.format(template, authkey, target))
    assert isinstance(obj, NotifyMSG91) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body=message_contents) is True
    assert mock_post.call_count == 1
    first_call = mock_post.call_args_list[0]
    assert first_call[0][0] == 'https://control.msg91.com/api/v5/flow/'
    response = loads(first_call[1]['data'])
    assert response['template_id'] == template
    assert response['short_url'] == 0
    assert len(response['recipients']) == 1
    assert response['recipients'][0]['mobiles'] == '15551233456'
    assert 'body' not in response['recipients'][0]
    assert 'type' not in response['recipients'][0]
    assert response['recipients'][0]['cat'] == 'info'