from unittest import mock
import requests
import pytest
from json import dumps, loads
from apprise import Apprise
from apprise.plugins.NotifyWhatsApp import NotifyWhatsApp
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('whatsapp://', {'instance': TypeError}), ('whatsapp://:@/', {'instance': TypeError}), ('whatsapp://{}@_'.format('a' * 32), {'instance': TypeError}), ('whatsapp://%20:{}@12345/{}'.format('e' * 32, '4' * 11), {'instance': TypeError}), ('whatsapp://{}@{}'.format('b' * 32, 10 ** 9), {'instance': NotifyWhatsApp, 'notify_response': False}), ('whatsapp://{}:{}@{}/123/{}/abcd/'.format('a' * 32, 'b' * 32, '3' * 11, '9' * 15), {'instance': NotifyWhatsApp, 'notify_response': False}), ('whatsapp://{}@12345/{}'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'privacy_url': 'whatsapp://e...e@1...5/%2B44444444444/'}), ('whatsapp://template:{}@12345/{}'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'privacy_url': 'whatsapp://template:e...e@1...5/%2B44444444444/'}), ('whatsapp://template:{}@12345/{}?lang=fr_CA'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'privacy_url': 'whatsapp://template:e...e@1...5/%2B44444444444/'}), ('whatsapp://{}@12345/{}?template=template&lang=fr_CA'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'privacy_url': 'whatsapp://template:e...e@1...5/%2B44444444444/'}), ('whatsapp://template:{}@12345/{}?lang=1234'.format('e' * 32, '4' * 11), {'instance': TypeError}), ('whatsapp://template:{}@12345/{}?:1=test&:body=3&:type=2'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'privacy_url': 'whatsapp://template:e...e@1...5/%2B44444444444/'}), ('whatsapp://template:{}@12345/{}?:invalid=23'.format('e' * 32, '4' * 11), {'instance': TypeError}), ('whatsapp://template:{}@12345/{}?:body='.format('e' * 32, '4' * 11), {'instance': TypeError}), ('whatsapp://template:{}@12345/{}?:1=Test&:body=1'.format('e' * 32, '4' * 11), {'instance': TypeError}), ('whatsapp://{}:{}@123456/{}'.format('a' * 32, 'b' * 32, '4' * 11), {'instance': NotifyWhatsApp}), ('whatsapp://_?token={}&from={}&to={}'.format('d' * 32, '5' * 11, '6' * 11), {'instance': NotifyWhatsApp}), ('whatsapp://_?token={}&source={}&to={}'.format('d' * 32, '5' * 11, '6' * 11), {'instance': NotifyWhatsApp}), ('whatsapp://{}@12345/{}'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'response': False, 'requests_response_code': 999}), ('whatsapp://{}@12345/{}'.format('e' * 32, '4' * 11), {'instance': NotifyWhatsApp, 'test_requests_exceptions': True}))

def test_plugin_whatsapp_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyWhatsApp() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_whatsapp_auth(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyWhatsApp() Auth\n      - account-wide auth token\n      - API key and its own auth token\n\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    token = '{}'.format('b' * 32)
    from_phone_id = '123456787654321'
    target = '+1 (555) 987-6543'
    message_contents = 'test'
    obj = Apprise.instantiate('whatsapp://{}@{}/{}'.format(token, from_phone_id, target))
    assert isinstance(obj, NotifyWhatsApp) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body=message_contents) is True
    assert mock_post.call_count == 1
    first_call = mock_post.call_args_list[0]
    assert first_call[0][0] == 'https://graph.facebook.com/v17.0/{}/messages'.format(from_phone_id)
    response = loads(first_call[1]['data'])
    assert response['text']['body'] == message_contents
    assert response['to'] == '+15559876543'
    assert response['recipient_type'] == 'individual'

@mock.patch('requests.post')
def test_plugin_whatsapp_edge_cases(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyWhatsApp() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    token = 'b' * 32
    from_phone_id = '123456787654321'
    targets = ('+1 (555) 123-3456',)
    with pytest.raises(TypeError):
        NotifyWhatsApp(token=None, from_phone_id=from_phone_id, targets=targets)
    with pytest.raises(TypeError):
        NotifyWhatsApp(token=token, from_phone_id=None, targets=targets)
    response.status_code = 400
    response.content = dumps({'error': {'code': 21211, 'message': "The 'To' number +1234567 is not a valid phone number."}})
    mock_post.return_value = response
    obj = NotifyWhatsApp(token=token, from_phone_id=from_phone_id, targets=targets)
    assert obj.notify('title', 'body', 'info') is False