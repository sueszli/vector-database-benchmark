from unittest import mock
import requests
from json import loads
from apprise import Apprise
from apprise.plugins.NotifyBulkSMS import NotifyBulkSMS
from helpers import AppriseURLTester
from apprise import NotifyType
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('bulksms://', {'instance': NotifyBulkSMS, 'notify_response': False}), ('bulksms://:@/', {'instance': NotifyBulkSMS, 'notify_response': False}), ('bulksms://{}@12345678'.format('a' * 10), {'instance': NotifyBulkSMS, 'notify_response': False}), ('bulksms://{}:{}@{}'.format('a' * 10, 'b' * 10, '3' * 5), {'instance': NotifyBulkSMS, 'notify_response': False}), ('bulksms://{}:{}@123/{}/abcd/'.format('a' * 5, 'b' * 10, '3' * 11), {'instance': NotifyBulkSMS, 'privacy_url': 'bulksms://a...a:****@+33333333333/@abcd'}), ('bulksms://{}:{}@{}?batch=y&unicode=n'.format('b' * 5, 'c' * 10, '4' * 11), {'instance': NotifyBulkSMS, 'privacy_url': 'bulksms://b...b:****@+4444444444'}), ('bulksms://{}:{}@123456/{}'.format('a' * 10, 'b' * 10, '4' * 11), {'instance': NotifyBulkSMS}), ('bulksms://{}:{}@{}'.format('a' * 10, 'b' * 10, '5' * 11), {'instance': NotifyBulkSMS}), ('bulksms://{}:{}@admin?route=premium'.format('a' * 10, 'b' * 10), {'instance': NotifyBulkSMS}), ('bulksms://{}:{}@admin?route=invalid'.format('a' * 10, 'b' * 10), {'instance': TypeError}), ('bulksms://_?user={}&password={}&from={}'.format('a' * 10, 'b' * 10, '5' * 11), {'instance': NotifyBulkSMS}), ('bulksms://_?user={}&password={}&from={}'.format('a' * 10, 'b' * 10, '5' * 3), {'instance': TypeError}), ('bulksms://_?user={}&password={}&from={}&to={}'.format('a' * 10, 'b' * 10, '5' * 11, '7' * 13), {'instance': NotifyBulkSMS}), ('bulksms://{}:{}@{}'.format('a' * 10, 'b' * 10, 'a' * 3), {'instance': NotifyBulkSMS, 'response': False, 'requests_response_code': 999}), ('bulksms://{}:{}@{}'.format('a' * 10, 'b' * 10, '6' * 11), {'instance': NotifyBulkSMS, 'test_requests_exceptions': True}))

def test_plugin_bulksms_urls():
    if False:
        print('Hello World!')
    '\n    NotifyTemplate() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_bulksms_edge_cases(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyBulkSMS() Edge Cases\n\n    '
    user = 'abcd'
    pwd = 'mypass123'
    targets = ['+1(555) 123-1234', '1555 5555555', 'group', '12', '@12']
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('bulksms://{}:{}@{}?batch=n'.format(user, pwd, '/'.join(targets)))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert len(obj) == 4
    assert mock_post.call_count == 4
    details = mock_post.call_args_list[0]
    payload = loads(details[1]['data'])
    assert payload['to'] == '+15551231234'
    assert payload['body'] == 'title\r\nbody'
    details = mock_post.call_args_list[1]
    payload = loads(details[1]['data'])
    assert payload['to'] == '+15555555555'
    assert payload['body'] == 'title\r\nbody'
    details = mock_post.call_args_list[2]
    payload = loads(details[1]['data'])
    assert isinstance(payload['to'], dict)
    assert payload['to']['name'] == 'group'
    assert payload['body'] == 'title\r\nbody'
    details = mock_post.call_args_list[3]
    payload = loads(details[1]['data'])
    assert isinstance(payload['to'], dict)
    assert payload['to']['name'] == '12'
    assert payload['body'] == 'title\r\nbody'
    assert obj.url().startswith('bulksms://{}:{}@{}'.format(user, pwd, '/'.join(['+15551231234', '+15555555555', '@group', '@12'])))
    assert 'batch=no' in obj.url()
    obj = Apprise.instantiate('bulksms://{}:{}@{}?batch=y'.format(user, pwd, '/'.join(targets)))
    assert len(obj) == 3