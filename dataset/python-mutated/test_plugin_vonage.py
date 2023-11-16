from unittest import mock
import pytest
import requests
from json import dumps
from apprise.plugins.NotifyVonage import NotifyVonage
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('vonage://', {'instance': TypeError}), ('vonage://:@/', {'instance': TypeError}), ('vonage://AC{}@12345678'.format('a' * 8), {'instance': TypeError}), ('vonage://AC{}:{}@{}'.format('a' * 8, 'b' * 16, '3' * 9), {'instance': TypeError}), ('vonage://AC{}:{}@{}/?ttl=0'.format('b' * 8, 'c' * 16, '3' * 11), {'instance': TypeError}), ('vonage://AC{}:{}@{}'.format('d' * 8, 'e' * 16, 'a' * 11), {'instance': TypeError}), ('vonage://AC{}:{}@{}/123/{}/abcd/'.format('f' * 8, 'g' * 16, '3' * 11, '9' * 15), {'instance': NotifyVonage, 'privacy_url': 'vonage://A...f:****@'}), ('vonage://AC{}:{}@{}'.format('h' * 8, 'i' * 16, '5' * 11), {'instance': NotifyVonage}), ('vonage://_?key=AC{}&secret={}&from={}'.format('a' * 8, 'b' * 16, '5' * 11), {'instance': NotifyVonage}), ('vonage://_?key=AC{}&secret={}&source={}'.format('a' * 8, 'b' * 16, '5' * 11), {'instance': NotifyVonage}), ('vonage://_?key=AC{}&secret={}&from={}&to={}'.format('a' * 8, 'b' * 16, '5' * 11, '7' * 13), {'instance': NotifyVonage}), ('vonage://AC{}:{}@{}'.format('a' * 8, 'b' * 16, '6' * 11), {'instance': NotifyVonage, 'response': False, 'requests_response_code': 999}), ('vonage://AC{}:{}@{}'.format('a' * 8, 'b' * 16, '6' * 11), {'instance': NotifyVonage, 'test_requests_exceptions': True}), ('nexmo://', {'instance': TypeError}), ('nexmo://:@/', {'instance': TypeError}), ('nexmo://AC{}@12345678'.format('a' * 8), {'instance': TypeError}), ('nexmo://AC{}:{}@{}'.format('a' * 8, 'b' * 16, '3' * 9), {'instance': TypeError}), ('nexmo://AC{}:{}@{}/?ttl=0'.format('b' * 8, 'c' * 16, '3' * 11), {'instance': TypeError}), ('nexmo://AC{}:{}@{}'.format('d' * 8, 'e' * 16, 'a' * 11), {'instance': TypeError}), ('nexmo://AC{}:{}@{}/123/{}/abcd/'.format('f' * 8, 'g' * 16, '3' * 11, '9' * 15), {'instance': NotifyVonage, 'privacy_url': 'vonage://A...f:****@'}), ('nexmo://AC{}:{}@{}'.format('h' * 8, 'i' * 16, '5' * 11), {'instance': NotifyVonage}), ('nexmo://_?key=AC{}&secret={}&from={}'.format('a' * 8, 'b' * 16, '5' * 11), {'instance': NotifyVonage}), ('nexmo://_?key=AC{}&secret={}&source={}'.format('a' * 8, 'b' * 16, '5' * 11), {'instance': NotifyVonage}), ('nexmo://_?key=AC{}&secret={}&from={}&to={}'.format('a' * 8, 'b' * 16, '5' * 11, '7' * 13), {'instance': NotifyVonage}), ('nexmo://AC{}:{}@{}'.format('a' * 8, 'b' * 16, '6' * 11), {'instance': NotifyVonage, 'response': False, 'requests_response_code': 999}), ('nexmo://AC{}:{}@{}'.format('a' * 8, 'b' * 16, '6' * 11), {'instance': NotifyVonage, 'test_requests_exceptions': True}))

def test_plugin_vonage_urls():
    if False:
        print('Hello World!')
    '\n    NotifyVonage() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_vonage_edge_cases(mock_post):
    if False:
        return 10
    '\n    NotifyVonage() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    apikey = 'AC{}'.format('b' * 8)
    secret = '{}'.format('b' * 16)
    source = '+1 (555) 123-3456'
    with pytest.raises(TypeError):
        NotifyVonage(apikey=None, secret=secret, source=source)
    with pytest.raises(TypeError):
        NotifyVonage(apikey='  ', secret=secret, source=source)
    with pytest.raises(TypeError):
        NotifyVonage(apikey=apikey, secret=None, source=source)
    with pytest.raises(TypeError):
        NotifyVonage(apikey=apikey, secret='  ', source=source)
    response.status_code = 400
    response.content = dumps({'code': 21211, 'message': "The 'To' number +1234567 is not a valid phone number."})
    mock_post.return_value = response
    obj = NotifyVonage(apikey=apikey, secret=secret, source=source)
    assert obj.notify('title', 'body', 'info') is False