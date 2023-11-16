from unittest import mock
import pytest
import requests
from json import dumps
from apprise.plugins.NotifySinch import NotifySinch
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('sinch://', {'instance': TypeError}), ('sinch://:@/', {'instance': TypeError}), ('sinch://{}@12345678'.format('a' * 32), {'instance': TypeError}), ('sinch://{}:{}@_'.format('a' * 32, 'b' * 32), {'instance': TypeError}), ('sinch://{}:{}@{}'.format('a' * 32, 'b' * 32, '3' * 5), {'instance': NotifySinch, 'notify_response': False}), ('sinch://{}:{}@{}'.format('a' * 32, 'b' * 32, '3' * 9), {'instance': TypeError}), ('sinch://{}:{}@{}/123/{}/abcd/'.format('a' * 32, 'b' * 32, '3' * 11, '9' * 15), {'instance': NotifySinch}), ('sinch://{}:{}@12345/{}'.format('a' * 32, 'b' * 32, '4' * 11), {'instance': NotifySinch, 'privacy_url': 'sinch://...aaaa:b...b@12345'}), ('sinch://{}:{}@123456/{}'.format('a' * 32, 'b' * 32, '4' * 11), {'instance': NotifySinch}), ('sinch://{}:{}@{}'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifySinch}), ('sinch://{}:{}@{}?region=eu'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifySinch}), ('sinch://{}:{}@{}?region=invalid'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': TypeError}), ('sinch://_?spi={}&token={}&from={}'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifySinch}), ('sinch://_?spi={}&token={}&source={}'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifySinch}), ('sinch://_?spi={}&token={}&from={}&to={}'.format('a' * 32, 'b' * 32, '5' * 11, '7' * 13), {'instance': NotifySinch}), ('sinch://{}:{}@{}'.format('a' * 32, 'b' * 32, '6' * 11), {'instance': NotifySinch, 'response': False, 'requests_response_code': 999}), ('sinch://{}:{}@{}'.format('a' * 32, 'b' * 32, '6' * 11), {'instance': NotifySinch, 'test_requests_exceptions': True}))

def test_plugin_sinch_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTemplate() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_sinch_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySinch() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    service_plan_id = '{}'.format('b' * 32)
    api_token = '{}'.format('b' * 32)
    source = '+1 (555) 123-3456'
    with pytest.raises(TypeError):
        NotifySinch(service_plan_id=None, api_token=api_token, source=source)
    with pytest.raises(TypeError):
        NotifySinch(service_plan_id=service_plan_id, api_token=None, source=source)
    response.status_code = 400
    response.content = dumps({'code': 21211, 'message': "The 'To' number +1234567 is not a valid phone number."})
    mock_post.return_value = response
    obj = NotifySinch(service_plan_id=service_plan_id, api_token=api_token, source=source)
    assert obj.notify('title', 'body', 'info') is False