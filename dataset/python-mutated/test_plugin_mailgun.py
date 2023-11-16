import os
from unittest import mock
import requests
from apprise.plugins.NotifyMailgun import NotifyMailgun
from helpers import AppriseURLTester
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('mailgun://', {'instance': TypeError}), ('mailgun://:@/', {'instance': TypeError}), ('mailgun://user@localhost.localdomain', {'instance': TypeError}), ('mailgun://localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': TypeError}), ('mailgun://"@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': TypeError}), ('mailgun://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?format=markdown'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?format=html'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?format=text'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?region=uS'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?region=EU'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?region=invalid'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': TypeError}), ('mailgun://user@localhost.localdomain/{}-{}-{}?from=jack@gmail.com&name=Jason<jason@gmail.com>'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?+X-Customer-Campaign-ID=Apprise'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?:name=Chris&:status=admin'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?:from=Chris&:status=admin'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?bcc=user@example.com&cc=user2@example.com'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}/test@example.com'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}?to=test@example.com'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}/test@example.com?name="Frodo"'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}/invalid'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun, 'notify_response': False}), ('mailgun://user@example.com/{}-{}-{}/{}?bcc={}&cc={}'.format('a' * 32, 'b' * 8, 'c' * 8, '/'.join(('user1@example.com', 'invalid', 'User2:user2@example.com')), ','.join(('user3@example.com', 'i@v', 'User1:user1@example.com')), ','.join(('user4@example.com', 'g@r@b', 'Da:user5@example.com'))), {'instance': NotifyMailgun}), ('mailgun://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('mailgun://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun, 'response': False, 'requests_response_code': 999}), ('mailgun://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifyMailgun, 'test_requests_exceptions': True}))

def test_plugin_mailgun_urls():
    if False:
        print('Hello World!')
    '\n    NotifyMailgun() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_mailgun_attachments(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyMailgun() Attachments\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = ''
    mock_post.return_value = okay_response
    apikey = 'abc123'
    obj = Apprise.instantiate('mailgun://user@localhost.localdomain/{}'.format(apikey))
    assert isinstance(obj, NotifyMailgun)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    mock_post.return_value = None
    mock_post.side_effect = OSError()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    mock_post.side_effect = None
    mock_post.return_value = okay_response
    with mock.patch('builtins.open', side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    with mock.patch('builtins.open', side_effect=(mock.Mock(), mock.Mock(), OSError())):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    with mock.patch('builtins.open') as mock_open:
        mock_fp = mock.Mock()
        mock_fp.seek.side_effect = OSError()
        mock_open.return_value = mock_fp
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
        mock_post.reset_mock()
        mock_fp.seek.side_effect = (None, None, OSError())
        mock_open.return_value = mock_fp
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    obj = Apprise.instantiate('mailgun://no-reply@example.com/{}/user1@example.com/user2@example.com?batch=yes'.format(apikey))
    assert isinstance(obj, NotifyMailgun)
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

@mock.patch('requests.post')
def test_plugin_mailgun_header_check(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyMailgun() Test Header Prep\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = ''
    mock_post.return_value = okay_response
    apikey = 'abc123'
    obj = Apprise.instantiate('mailgun://user@localhost.localdomain/{}'.format(apikey))
    assert isinstance(obj, NotifyMailgun)
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.mailgun.net/v3/localhost.localdomain/messages'
    payload = mock_post.call_args_list[0][1]['data']
    assert 'from' in payload
    assert 'Apprise <user@localhost.localdomain>' == payload['from']
    assert 'user@localhost.localdomain' == payload['to']
    mock_post.reset_mock()
    obj = Apprise.instantiate('mailgun://user@localhost.localdomain/{}?from=Luke%20Skywalker'.format(apikey))
    assert isinstance(obj, NotifyMailgun)
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    payload = mock_post.call_args_list[0][1]['data']
    assert 'from' in payload
    assert 'to' in payload
    assert 'Luke Skywalker <user@localhost.localdomain>' == payload['from']
    assert 'user@localhost.localdomain' == payload['to']
    mock_post.reset_mock()
    obj = Apprise.instantiate('mailgun://user@localhost.localdomain/{}?from=Luke%20Skywalker<luke@rebels.com>'.format(apikey))
    assert isinstance(obj, NotifyMailgun)
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    payload = mock_post.call_args_list[0][1]['data']
    assert 'from' in payload
    assert 'to' in payload
    assert 'Luke Skywalker <luke@rebels.com>' == payload['from']
    assert 'luke@rebels.com' == payload['to']
    mock_post.reset_mock()
    obj = Apprise.instantiate('mailgun://user@localhost.localdomain/{}?from=luke@rebels.com'.format(apikey))
    assert isinstance(obj, NotifyMailgun)
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    payload = mock_post.call_args_list[0][1]['data']
    assert 'from' in payload
    assert 'to' in payload
    assert 'luke@rebels.com' == payload['from']
    assert 'luke@rebels.com' == payload['to']