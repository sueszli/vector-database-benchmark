import os
from unittest import mock
import requests
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise.plugins.NotifySMTP2Go import NotifySMTP2Go
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('smtp2go://', {'instance': TypeError}), ('smtp2go://:@/', {'instance': TypeError}), ('smtp2go://user@localhost.localdomain', {'instance': TypeError}), ('smtp2go://localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': TypeError}), ('smtp2go://"@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': TypeError}), ('smtp2go://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}?format=markdown'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}?format=html'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}?format=text'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}?+X-Customer-Campaign-ID=Apprise'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}?bcc=user@example.com&cc=user2@example.com'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}/test@example.com'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}?to=test@example.com'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}/test@example.com?name="Frodo"'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}/invalid'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go, 'notify_response': False}), ('smtp2go://user@example.com/{}-{}-{}/{}?bcc={}&cc={}'.format('a' * 32, 'b' * 8, 'c' * 8, '/'.join(('user1@example.com', 'invalid', 'User2:user2@example.com')), ','.join(('user3@example.com', 'i@v', 'User1:user1@example.com')), ','.join(('user4@example.com', 'g@r@b', 'Da:user5@example.com'))), {'instance': NotifySMTP2Go}), ('smtp2go://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('smtp2go://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go, 'response': False, 'requests_response_code': 999}), ('smtp2go://user@localhost.localdomain/{}-{}-{}'.format('a' * 32, 'b' * 8, 'c' * 8), {'instance': NotifySMTP2Go, 'test_requests_exceptions': True}))

def test_plugin_smtp2go_urls():
    if False:
        return 10
    '\n    NotifySMTP2Go() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_smtp2go_attachments(mock_post):
    if False:
        return 10
    '\n    NotifySMTP2Go() Attachments\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = ''
    mock_post.return_value = okay_response
    apikey = 'abc123'
    obj = Apprise.instantiate('smtp2go://user@localhost.localdomain/{}'.format(apikey))
    assert isinstance(obj, NotifySMTP2Go)
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
    obj = Apprise.instantiate('smtp2go://no-reply@example.com/{}/user1@example.com/user2@example.com?batch=yes'.format(apikey))
    assert isinstance(obj, NotifySMTP2Go)
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