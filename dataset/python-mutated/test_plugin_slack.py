import os
from unittest import mock
from inspect import cleandoc
import pytest
import requests
from apprise import Apprise
from apprise import NotifyType
from apprise import AppriseAttachment
from apprise.plugins.NotifySlack import NotifySlack
from helpers import AppriseURLTester
from json import loads, dumps
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('slack://', {'instance': TypeError}), ('slack://:@/', {'instance': TypeError}), ('slack://T1JJ3T3L2', {'instance': TypeError}), ('slack://T1JJ3T3L2/A1BRTD4JD/', {'instance': TypeError}), ('slack://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#hmm/#-invalid-', {'instance': NotifySlack, 'response': False, 'requests_response_text': {'ok': False, 'message': 'Bad Channel'}}), ('slack://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#channel', {'instance': NotifySlack, 'include_image': False, 'requests_response_text': 'ok'}), ('slack://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/+id/@id/', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://username@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/?to=#nuxref', {'instance': NotifySlack, 'privacy_url': 'slack://username@T...2/A...D/T...Q/', 'requests_response_text': 'ok'}), ('slack://username@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#nuxref', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://T1JJ3T3L2/A1BRTD4JD/TIiajkdnl/user@gmail.com', {'instance': NotifySlack, 'requests_response_text': 'ok', 'notify_response': False}), ('slack://bot@_/#nuxref?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnadfdajkjkfl/', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/&to=#chan', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://username@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#nuxref', {'instance': NotifySlack, 'requests_response_text': 'fail', 'notify_response': False}), ('slack://username@xoxb-1234-1234-abc124/#nuxref?footer=no', {'instance': NotifySlack, 'requests_response_text': {'ok': True, 'message': '', 'file': {'url_private': 'http://localhost/'}}}), ('slack://?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/&to=#chan&blocks=yes&footer=yes', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/&to=#chan&blocks=yes&footer=no', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/&to=#chan&blocks=yes&footer=yes&image=no', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/&to=#chan&blocks=yes&format=text', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://?token=T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/&to=#chan&blocks=no&format=text', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://?token=xoxb-1234-1234-abc124&to=#nuxref&footer=no&user=test', {'instance': NotifySlack, 'requests_response_text': {'ok': True, 'message': '', 'file': {'url_private': 'http://localhost/'}}, 'privacy_url': 'slack://test@x...4/nuxref/'}), ('slack://?token=xoxb-1234-1234-abc124&to=#nuxref,#$,#-&footer=no', {'instance': NotifySlack, 'requests_response_text': {'ok': True, 'message': '', 'file': {'url_private': 'http://localhost/'}}, 'notify_response': False}), ('slack://username@xoxb-1234-1234-abc124/#nuxref', {'instance': NotifySlack, 'requests_response_text': {'ok': True, 'message': ''}, 'response': False}), ('slack://username@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ', {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('https://hooks.slack.com/services/{}/{}/{}'.format('A' * 9, 'B' * 9, 'c' * 24), {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('https://hooks.slack.com/services/{}/{}/{}?format=text'.format('A' * 9, 'B' * 9, 'c' * 24), {'instance': NotifySlack, 'requests_response_text': 'ok'}), ('slack://username@-INVALID-/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#cool', {'instance': TypeError}), ('slack://username@T1JJ3T3L2/-INVALID-/TIiajkdnlazkcOXrIdevi7FQ/#great', {'instance': TypeError}), ('slack://username@T1JJ3T3L2/A1BRTD4JD/-INVALID-/#channel', {'instance': TypeError}), ('slack://l2g@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#usenet', {'instance': NotifySlack, 'response': False, 'requests_response_code': requests.codes.internal_server_error, 'requests_response_text': 'ok'}), ('slack://respect@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#a', {'instance': NotifySlack, 'response': False, 'requests_response_code': 999, 'requests_response_text': 'ok'}), ('slack://notify@T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#b', {'instance': NotifySlack, 'test_requests_exceptions': True, 'requests_response_text': 'ok'}))

def test_plugin_slack_urls():
    if False:
        while True:
            i = 10
    '\n    NotifySlack() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_slack_oauth_access_token(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySlack() OAuth Access Token Tests\n\n    '
    token = 'xo-invalid'
    request = mock.Mock()
    request.content = dumps({'ok': True, 'message': '', 'file': {'url_private': 'http://localhost'}})
    request.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifySlack(access_token=token)
    token = 'xoxb-1234-1234-abc124'
    mock_post.return_value = request
    obj = NotifySlack(access_token=token, targets='#apprise')
    assert isinstance(obj, NotifySlack) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    mock_post.reset_mock()
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://slack.com/api/chat.postMessage'
    assert mock_post.call_args_list[1][0][0] == 'https://slack.com/api/files.upload'
    mock_post.return_value = None
    mock_post.side_effect = (request, requests.ConnectionError(0, 'requests.ConnectionError() not handled'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    mock_post.return_value = None
    mock_post.side_effect = (request, OSError(0, 'OSError'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    mock_post.return_value = request
    mock_post.side_effect = None
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    request.content = dumps({'ok': True, 'message': '', 'file': None})
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    request.content = '{'
    assert obj.send(body='test', attach=attach) is False
    request.content = dumps({'ok': False, 'message': 'We failed'})
    assert obj.send(body='test', attach=attach) is False
    mock_post.side_effect = OSError('Attachment Error')
    mock_post.return_value = None
    assert obj.send(body='test') is False

@mock.patch('requests.post')
def test_plugin_slack_webhook_mode(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifySlack() Webhook Mode Tests\n\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = b'ok'
    mock_post.return_value.text = 'ok'
    token_a = 'A' * 9
    token_b = 'B' * 9
    token_c = 'c' * 24
    channels = 'chan1,#chan2,+BAK4K23G5,@user,,,'
    obj = NotifySlack(token_a=token_a, token_b=token_b, token_c=token_c, targets=channels)
    assert len(obj.channels) == 4
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    with pytest.raises(TypeError):
        NotifySlack(token_a=None, token_b=token_b, token_c=token_c, targets=channels)
    obj = NotifySlack(token_a=token_a, token_b=token_b, token_c=token_c, targets=channels, include_image=True)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_slack_send_by_email(mock_get, mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySlack() Send by Email Tests\n\n    '
    token = 'xoxb-1234-1234-abc124'
    request = mock.Mock()
    request.content = dumps({'ok': True, 'message': '', 'user': {'id': 'ABCD1234'}})
    request.status_code = requests.codes.ok
    mock_post.return_value = request
    mock_get.return_value = request
    obj = NotifySlack(access_token=token, targets='user@gmail.com')
    assert isinstance(obj, NotifySlack) is True
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_get.call_count == 1
    assert mock_post.call_count == 1
    assert mock_get.call_args_list[0][0][0] == 'https://slack.com/api/users.lookupByEmail'
    assert mock_post.call_args_list[0][0][0] == 'https://slack.com/api/chat.postMessage'
    mock_post.reset_mock()
    mock_get.reset_mock()
    mock_post.return_value = request
    mock_get.return_value = request
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_get.call_count == 0
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://slack.com/api/chat.postMessage'
    request.content = dumps({'ok': False, 'message': ''})
    mock_post.reset_mock()
    mock_get.reset_mock()
    mock_post.return_value = request
    mock_get.return_value = request
    obj = NotifySlack(access_token=token, targets='user@gmail.com')
    assert isinstance(obj, NotifySlack) is True
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert mock_get.call_count == 1
    assert mock_post.call_count == 0
    assert mock_get.call_args_list[0][0][0] == 'https://slack.com/api/users.lookupByEmail'
    request.content = '}'
    mock_post.reset_mock()
    mock_get.reset_mock()
    mock_post.return_value = request
    mock_get.return_value = request
    obj = NotifySlack(access_token=token, targets='user@gmail.com')
    assert isinstance(obj, NotifySlack) is True
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert mock_get.call_count == 1
    assert mock_post.call_count == 0
    assert mock_get.call_args_list[0][0][0] == 'https://slack.com/api/users.lookupByEmail'
    request.content = '}'
    mock_post.reset_mock()
    mock_get.reset_mock()
    mock_post.return_value = request
    mock_get.return_value = request
    obj = NotifySlack(access_token=token, targets='user@gmail.com')
    assert isinstance(obj, NotifySlack) is True
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert mock_get.call_count == 1
    assert mock_post.call_count == 0
    assert mock_get.call_args_list[0][0][0] == 'https://slack.com/api/users.lookupByEmail'
    request.content = dumps({'ok': True, 'message': '', 'user': {'id': 'ABCD1234'}})
    request.status_code = requests.codes.ok
    mock_post.reset_mock()
    mock_get.reset_mock()
    mock_post.return_value = request
    mock_get.side_effect = requests.ConnectionError(0, 'requests.ConnectionError() not handled')
    obj = NotifySlack(access_token=token, targets='user@gmail.com')
    assert isinstance(obj, NotifySlack) is True
    assert isinstance(obj.url(), str) is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 0
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert mock_get.call_count == 1
    assert mock_post.call_count == 0
    assert mock_get.call_args_list[0][0][0] == 'https://slack.com/api/users.lookupByEmail'

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_slack_markdown(mock_get, mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifySlack() Markdown tests\n\n    '
    request = mock.Mock()
    request.content = b'ok'
    request.status_code = requests.codes.ok
    mock_post.return_value = request
    mock_get.return_value = request
    aobj = Apprise()
    assert aobj.add('slack://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/#channel')
    body = cleandoc("\n    Here is a <https://slack.com|Slack Link> we want to support as part of it's\n    markdown.\n\n    This one has arguments we want to preserve:\n       <https://slack.com?arg=val&arg2=val2|Slack Link>.\n    We also want to be able to support <https://slack.com> links without the\n    description.\n\n    Channel Testing\n    <!channelA>\n    <!channelA|Description>\n    ")
    assert aobj.notify(body=body, title='title', notify_type=NotifyType.INFO)
    assert mock_get.call_count == 0
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://hooks.slack.com/services/T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ'
    data = loads(mock_post.call_args_list[0][1]['data'])
    assert data['attachments'][0]['text'] == "Here is a <https://slack.com|Slack Link> we want to support as part of it's\nmarkdown.\n\nThis one has arguments we want to preserve:\n   <https://slack.com?arg=val&arg2=val2|Slack Link>.\nWe also want to be able to support <https://slack.com> links without the\ndescription.\n\nChannel Testing\n<!channelA>\n<!channelA|Description>"