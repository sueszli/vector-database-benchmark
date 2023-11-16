import os
import pytest
from unittest import mock
import requests
from json import dumps
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise.plugins.NotifySparkPost import NotifySparkPost
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('sparkpost://', {'instance': TypeError}), ('sparkpost://:@/', {'instance': TypeError}), ('sparkpost://user@localhost.localdomain', {'instance': TypeError}), ('sparkpost://localhost.localdomain/{}'.format('a' * 32), {'instance': TypeError}), ('sparkpost://"@localhost.localdomain/{}'.format('b' * 32), {'instance': TypeError}), ('sparkpost://user@localhost.localdomain/{}'.format('c' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?format=markdown'.format('d' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?format=html'.format('d' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?format=text'.format('d' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?region=uS'.format('d' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?region=EU'.format('e' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?+X-Customer-Campaign-ID=Apprise'.format('f' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?:name=Chris&:status=admin'.format('g' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?bcc=user@example.com&cc=user2@example.com'.format('h' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?region=invalid'.format('a' * 32), {'instance': TypeError}), ('sparkpost://user@localhost.localdomain/{}/test@example.com'.format('a' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}/invalid'.format('i' * 32), {'instance': NotifySparkPost, 'notify_response': False}), ('sparkpost://user@example.com/{}/{}?bcc={}&cc={}'.format('j' * 32, '/'.join(('user1@example.com', 'invalid', 'User2:user2@example.com')), ','.join(('user3@example.com', 'i@v', 'User1:user1@example.com')), ','.join(('user4@example.com', 'g@r@b', 'Da:user5@example.com'))), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}?to=test@example.com'.format('k' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}/test@example.com?name="Frodo"'.format('l' * 32), {'instance': NotifySparkPost, 'requests_response_text': {'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}}}), ('sparkpost://user@localhost.localdomain/{}'.format('m' * 32), {'instance': NotifySparkPost, 'requests_response_text': '{'}), ('sparkpost://user@localhost.localdomain/{}'.format('n' * 32), {'instance': NotifySparkPost, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('sparkpost://user@localhost.localdomain/{}'.format('o' * 32), {'instance': NotifySparkPost, 'response': False, 'requests_response_code': 999}), ('sparkpost://user@localhost.localdomain/{}'.format('p' * 32), {'instance': NotifySparkPost, 'test_requests_exceptions': True}))

def test_plugin_sparkpost_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifySparkPost() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_sparkpost_throttling(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySparkPost() Throttling\n\n    '
    NotifySparkPost.sparkpost_retry_wait_sec = 0.1
    NotifySparkPost.sparkpost_retry_attempts = 3
    apikey = 'abc123'
    user = 'user'
    host = 'example.com'
    targets = '{}@{}'.format(user, host)
    with pytest.raises(TypeError):
        NotifySparkPost(apikey=apikey, targets=targets, host=host)
    with pytest.raises(TypeError):
        NotifySparkPost(apikey=None, targets=targets, user=user, host=host)
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = dumps({'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}})
    retry_response = requests.Request()
    retry_response.status_code = requests.codes.too_many_requests
    retry_response.content = dumps({'errors': [{'description': 'Unconfigured or unverified sending domain.', 'code': '7001', 'message': 'Invalid domain'}]})
    mock_post.side_effect = (retry_response, retry_response, okay_response)
    obj = Apprise.instantiate('sparkpost://user@localhost.localdomain/{}'.format(apikey))
    assert isinstance(obj, NotifySparkPost)
    assert obj.notify('test') is True
    mock_post.reset_mock()
    mock_post.side_effect = (retry_response, retry_response, retry_response)
    assert obj.notify('test') is False

@mock.patch('requests.post')
def test_plugin_sparkpost_attachments(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifySparkPost() Attachments\n\n    '
    NotifySparkPost.sparkpost_retry_wait_sec = 0.1
    NotifySparkPost.sparkpost_retry_attempts = 3
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = dumps({'results': {'total_rejected_recipients': 0, 'total_accepted_recipients': 1, 'id': '11668787484950529'}})
    mock_post.return_value = okay_response
    apikey = 'abc123'
    obj = Apprise.instantiate('sparkpost://user@localhost.localdomain/{}'.format(apikey))
    assert isinstance(obj, NotifySparkPost)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    with mock.patch('base64.b64encode', side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    obj = Apprise.instantiate('sparkpost://no-reply@example.com/{}/user1@example.com/user2@example.com?batch=yes'.format(apikey))
    assert isinstance(obj, NotifySparkPost)
    assert len(obj) == 1
    obj.default_batch_size = 1
    assert len(obj) == 2
    mock_post.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 2
    mock_post.reset_mock()
    obj.default_batch_size = 2
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 1