import os
from unittest import mock
from apprise.plugins.NotifyAppriseAPI import NotifyAppriseAPI
from helpers import AppriseURLTester
import requests
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('apprise://', {'instance': None}), ('apprise://:@/', {'instance': None}), ('apprise://localhost', {'instance': TypeError}), ('apprise://localhost/!', {'instance': TypeError}), ('apprise://localhost/%%20', {'instance': TypeError}), ('apprise://localhost/%s' % ('a' * 32), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://localhost/a...a/'}), ('apprise://localhost:8080/%s' % ('b' * 32), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://localhost:8080/b...b/'}), ('apprises://localhost/%s' % ('c' * 32), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://localhost/c...c/'}), ('https://example.com/path/notify/%s' % ('d' * 32), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://example.com/path/d...d/'}), ('http://example.com/notify/%s' % ('d' * 32), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://example.com/d...d/'}), ('apprises://localhost/?to=%s' % ('e' * 32), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://localhost/e...e/'}), ('apprise://localhost/?token=%s&to=%s' % ('f' * 32, 'abcd'), {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://localhost/f...f/'}), ('apprise://localhost/?token=%s&tags=admin,team' % 'abcd', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://localhost/a...d/'}), ('apprise://user@localhost/mytoken0/?format=markdown', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://user@localhost/m...0/'}), ('apprise://user@localhost/mytoken1/', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://user@localhost/m...1/'}), ('apprise://localhost:8080/mytoken/', {'instance': NotifyAppriseAPI}), ('apprise://user:pass@localhost:8080/mytoken2/', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprise://user:****@localhost:8080/m...2/'}), ('apprises://localhost/mytoken/', {'instance': NotifyAppriseAPI}), ('apprises://user:pass@localhost/mytoken3/', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://user:****@localhost/m...3/'}), ('apprises://localhost:8080/mytoken4/', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://localhost:8080/m...4/'}), ('apprises://localhost:8080/abc123/?method=json', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://localhost:8080/a...3/'}), ('apprises://localhost:8080/abc123/?method=form', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://localhost:8080/a...3/'}), ('apprises://localhost:8080/abc123/?method=invalid', {'instance': TypeError}), ('apprises://user:password@localhost:8080/mytoken5/', {'instance': NotifyAppriseAPI, 'privacy_url': 'apprises://user:****@localhost:8080/m...5/'}), ('apprises://localhost:8080/path?+HeaderKey=HeaderValue', {'instance': NotifyAppriseAPI}), ('apprise://localhost/%s' % ('a' * 32), {'instance': NotifyAppriseAPI, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('apprise://localhost/%s' % ('a' * 32), {'instance': NotifyAppriseAPI, 'response': False, 'requests_response_code': 999}), ('apprise://localhost/%s' % ('a' * 32), {'instance': NotifyAppriseAPI, 'test_requests_exceptions': True}))

def test_plugin_apprise_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyAppriseAPI() General Checks\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_notify_apprise_api_attachments(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyAppriseAPI() Attachments\n\n    '
    okay_response = requests.Request()
    for method in ('json', 'form'):
        okay_response.status_code = requests.codes.ok
        okay_response.content = ''
        mock_post.return_value = okay_response
        obj = Apprise.instantiate('apprise://user@localhost/mytoken1/?method={}'.format(method))
        assert isinstance(obj, NotifyAppriseAPI)
        path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
        attach = AppriseAttachment(path)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
        path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
        path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
        attach = AppriseAttachment(path)
        mock_post.side_effect = None
        mock_post.return_value = okay_response
        with mock.patch('builtins.open', side_effect=OSError()):
            assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
        with mock.patch('requests.post', side_effect=OSError()):
            assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
        obj = Apprise.instantiate('apprise://user@localhost/mytoken1/')
        assert isinstance(obj, NotifyAppriseAPI)
        mock_post.reset_mock()
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
        assert mock_post.call_count == 1
        details = mock_post.call_args_list[0]
        assert details[0][0] == 'http://localhost/notify/mytoken1'
        assert obj.url(privacy=False).startswith('apprise://user@localhost/mytoken1/')
        mock_post.reset_mock()