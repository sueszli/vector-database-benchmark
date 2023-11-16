import os
import pytest
from unittest import mock
import requests
from json import dumps
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise.plugins.NotifyPushSafer import NotifyPushSafer
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('psafer://:@/', {'instance': TypeError}), ('psafer://', {'instance': TypeError}), ('psafers://', {'instance': TypeError}), ('psafer://{}'.format('a' * 20), {'instance': NotifyPushSafer, 'notify_response': False}), ('psafer://{}'.format('b' * 20), {'instance': NotifyPushSafer, 'requests_response_text': '{', 'notify_response': False}), ('psafer://{}'.format('c' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 0, 'error': 'we failed'}, 'notify_response': False}), ('psafers://{}'.format('d' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 0}, 'notify_response': False}), ('psafer://{}'.format('e' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}}), ('psafer://{}/12/24/53'.format('e' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}}), ('psafer://{}?to=12,24,53'.format('e' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}}), ('psafer://{}?priority=emergency'.format('f' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}}), ('psafer://{}?priority=-1'.format('f' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}}), ('psafer://{}?priority=invalid'.format('f' * 20), {'instance': TypeError}), ('psafer://{}?priority=25'.format('f' * 20), {'instance': TypeError}), ('psafer://{}?sound=ok'.format('g' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}}), ('psafers://{}?sound=14'.format('g' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}, 'privacy_url': 'psafers://g...g'}), ('psafer://{}?sound=invalid'.format('h' * 20), {'instance': TypeError}), ('psafer://{}?sound=94000'.format('h' * 20), {'instance': TypeError}), ('psafers://{}?vibration=1'.format('h' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 1}, 'privacy_url': 'psafers://h...h'}), ('psafer://{}?vibration=invalid'.format('h' * 20), {'instance': TypeError}), ('psafer://{}?vibration=25000'.format('h' * 20), {'instance': TypeError}), ('psafers://{}'.format('d' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 0}, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('psafer://{}'.format('d' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 0}, 'response': False, 'requests_response_code': 999}), ('psafers://{}'.format('d' * 20), {'instance': NotifyPushSafer, 'requests_response_text': {'status': 0}, 'test_requests_exceptions': True}))

def test_plugin_pushsafer_urls():
    if False:
        print('Hello World!')
    '\n    NotifyPushSafer() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_pushsafer_general(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyPushSafer() General Tests\n\n    '
    privatekey = 'abc123'
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = dumps({'status': 1, 'success': 'okay'})
    with pytest.raises(TypeError):
        NotifyPushSafer(privatekey=None)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment()
    for _ in range(0, 4):
        attach.add(path)
    obj = NotifyPushSafer(privatekey=privatekey)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    with mock.patch('builtins.open', side_effect=OSError):
        obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach)
    attach = AppriseAttachment(path)
    attach[0]._mimetype = 'application/octet-stream'
    mock_post.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert 'data' in mock_post.call_args[1]
    assert 'p' not in mock_post.call_args[1]['data']
    assert 'p2' not in mock_post.call_args[1]['data']
    assert 'p3' not in mock_post.call_args[1]['data']
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False