import requests
from json import loads
from unittest import mock
from apprise.plugins.NotifyD7Networks import NotifyD7Networks
from helpers import AppriseURLTester
from apprise import Apprise
from apprise import NotifyType
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('d7sms://', {'instance': TypeError}), ('d7sms://:@/', {'instance': TypeError}), ('d7sms://token@{}/{}/{}'.format('1' * 9, '2' * 15, 'a' * 13), {'instance': NotifyD7Networks, 'notify_response': False}), ('d7sms://token1@{}?batch=yes'.format('3' * 14), {'instance': NotifyD7Networks, 'privacy_url': 'd7sms://t...1@'}), ('d7sms://token:colon2@{}?batch=yes'.format('3' * 14), {'instance': NotifyD7Networks, 'privacy_url': 'd7sms://t...2@'}), ('d7sms://:token3@{}?batch=yes'.format('3' * 14), {'instance': NotifyD7Networks, 'privacy_url': 'd7sms://:...3@'}), ('d7sms://{}?token=token6'.format('3' * 14), {'instance': NotifyD7Networks, 'privacy_url': 'd7sms://t...6@'}), ('d7sms://token4@{}?unicode=no'.format('3' * 14), {'instance': NotifyD7Networks, 'privacy_url': 'd7sms://t...4@'}), ('d7sms://token8@{}/{}/?unicode=yes'.format('3' * 14, '4' * 14), {'instance': NotifyD7Networks, 'privacy_url': 'd7sms://t...8@'}), ('d7sms://token@{}?batch=yes&to={}'.format('3' * 14, '6' * 14), {'instance': NotifyD7Networks}), ('d7sms://token@{}?batch=yes&from=apprise'.format('3' * 14), {'instance': NotifyD7Networks}), ('d7sms://token@{}?batch=yes&source=apprise'.format('3' * 14), {'instance': NotifyD7Networks}), ('d7sms://token@{}?batch=no'.format('3' * 14), {'instance': NotifyD7Networks}), ('d7sms://token@{}'.format('3' * 14), {'instance': NotifyD7Networks, 'response': False, 'requests_response_code': 999}), ('d7sms://token@{}'.format('3' * 14), {'instance': NotifyD7Networks, 'test_requests_exceptions': True}))

def test_plugin_d7networks_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyD7Networks() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_d7networks_edge_cases(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyD7Networks() Edge Cases tests\n\n    '
    request = mock.Mock()
    request.content = '{}'
    request.status_code = requests.codes.ok
    mock_post.return_value = request
    aobj = Apprise()
    assert aobj.add('d7sms://Token@15551231234/15551231236')
    body = 'test message'
    assert aobj.notify(body=body, title='title', notify_type=NotifyType.INFO)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.d7networks.com/messages/v1/send'
    assert mock_post.call_args_list[1][0][0] == 'https://api.d7networks.com/messages/v1/send'
    data = loads(mock_post.call_args_list[0][1]['data'])
    assert len(data['messages']) == 1
    message = data['messages'][0]
    assert len(message['recipients']) == 1
    assert message['content'] == 'title\r\ntest message'
    assert message['data_coding'] == 'auto'
    data = loads(mock_post.call_args_list[1][1]['data'])
    assert len(data['messages']) == 1
    message = data['messages'][0]
    assert len(message['recipients']) == 1
    assert message['content'] == 'title\r\ntest message'
    assert message['data_coding'] == 'auto'
    mock_post.reset_mock()
    aobj = Apprise()
    assert aobj.add('d7sms://Token@15551231234/15551231236?batch=yes')
    body = 'test message'
    assert aobj.notify(body=body, title='title', notify_type=NotifyType.INFO)
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.d7networks.com/messages/v1/send'
    data = loads(mock_post.call_args_list[0][1]['data'])
    assert len(data['messages']) == 1
    message = data['messages'][0]
    assert len(message['recipients']) == 2
    assert '15551231234' in message['recipients']
    assert '15551231236' in message['recipients']
    assert message['content'] == 'title\r\ntest message'
    assert message['data_coding'] == 'auto'