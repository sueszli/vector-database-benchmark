import os
from unittest import mock
from datetime import datetime, timedelta
from datetime import timezone
import pytest
import requests
from apprise.plugins.NotifyDiscord import NotifyDiscord
from helpers import AppriseURLTester
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise import NotifyFormat
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('discord://', {'instance': TypeError}), ('discord://:@/', {'instance': TypeError}), ('discord://%s' % ('i' * 24), {'instance': TypeError}), ('discord://%s/%s' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://l2g@%s/%s' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?format=markdown&footer=Yes&image=Yes' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content, 'include_image': False}), ('discord://%s/%s?format=markdown&footer=Yes&image=No&fields=no' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?format=markdown&footer=Yes&image=Yes' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('https://discord.com/api/webhooks/{}/{}'.format('0' * 10, 'B' * 40), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('https://discordapp.com/api/webhooks/{}/{}'.format('0' * 10, 'B' * 40), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('https://discordapp.com/api/webhooks/{}/{}?footer=yes'.format('0' * 10, 'B' * 40), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?format=markdown&avatar=No&footer=No' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?format=markdown' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?format=markdown&thread=abc123' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?format=text' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?hmarkdown=true&ref=http://localhost' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?markdown=true&url=http://localhost' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s?avatar_url=http://localhost/test.jpg' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content}), ('discord://%s/%s' % ('i' * 24, 't' * 64), {'instance': NotifyDiscord, 'requests_response_code': requests.codes.no_content, 'include_image': False}), ('discord://%s/%s/' % ('a' * 24, 'b' * 64), {'instance': NotifyDiscord, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('discord://%s/%s/' % ('a' * 24, 'b' * 64), {'instance': NotifyDiscord, 'response': False, 'requests_response_code': 999}), ('discord://%s/%s/' % ('a' * 24, 'b' * 64), {'instance': NotifyDiscord, 'test_requests_exceptions': True}))

def test_plugin_discord_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyDiscord() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_discord_general(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyDiscord() General Checks\n\n    '
    NotifyDiscord.clock_skew = timedelta(seconds=0)
    epoch = datetime.fromtimestamp(0, timezone.utc)
    webhook_id = 'A' * 24
    webhook_token = 'B' * 64
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = ''
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    with pytest.raises(TypeError):
        NotifyDiscord(webhook_id=None, webhook_token=webhook_token)
    with pytest.raises(TypeError):
        NotifyDiscord(webhook_id='  ', webhook_token=webhook_token)
    with pytest.raises(TypeError):
        NotifyDiscord(webhook_id=webhook_id, webhook_token=None)
    with pytest.raises(TypeError):
        NotifyDiscord(webhook_id=webhook_id, webhook_token='   ')
    obj = NotifyDiscord(webhook_id=webhook_id, webhook_token=webhook_token, footer=True, thumbnail=False)
    assert obj.ratelimit_remaining == 1
    assert isinstance(obj.url(), str) is True
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 0}
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 0
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 10}
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    del mock_post.return_value.headers['X-RateLimit-Reset']
    assert obj.send(body='test') is True
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds() + 1, 'X-RateLimit-Remaining': 0}
    obj.ratelimit_remaining = 0
    assert obj.send(body='test') is True
    mock_post.return_value.status_code = requests.codes.too_many_requests
    assert obj.send(body='test') is False
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds() - 1, 'X-RateLimit-Remaining': 0}
    assert obj.send(body='test') is True
    obj.ratelimit_remaining = 1
    mock_post.return_value.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    test_markdown = 'body'
    (desc, results) = obj.extract_markdown_sections(test_markdown)
    assert isinstance(results, list) is True
    assert len(results) == 0
    test_markdown = '\n    A section of text that has no header at the top.\n    It also has a hash tag # <- in the middle of a\n    string.\n\n    ## Heading 1\n    body\n\n    # Heading 2\n\n    more content\n    on multi-lines\n    '
    (desc, results) = obj.extract_markdown_sections(test_markdown)
    assert isinstance(desc, str) is True
    assert desc.startswith('A section of text that has no header at the top.')
    assert desc.endswith('string.')
    assert isinstance(results, list) is True
    assert len(results) == 2
    assert results[0]['name'] == 'Heading 1'
    assert results[0]['value'] == '```md\nbody\n```'
    assert results[1]['name'] == 'Heading 2'
    assert results[1]['value'] == '```md\nmore content\n    on multi-lines\n```'
    test_markdown = '## Heading one\nbody body\n\n' + '# Heading 2 ##\n\nTest\n\n' + 'more content\n' + 'even more content  \t\r\n\n\n' + '# Heading 3 ##\n\n\n' + 'normal content\n' + '# heading 4\n' + '#### Heading 5'
    (desc, results) = obj.extract_markdown_sections(test_markdown)
    assert isinstance(results, list) is True
    assert isinstance(desc, str) is True
    assert not desc
    assert len(results) == 5
    assert results[0]['name'] == 'Heading one'
    assert results[0]['value'] == '```md\nbody body\n```'
    assert results[1]['name'] == 'Heading 2'
    assert results[1]['value'] == '```md\nTest\n\nmore content\neven more content\n```'
    assert results[2]['name'] == 'Heading 3'
    assert results[2]['value'] == '```md\nnormal content\n```'
    assert results[3]['name'] == 'heading 4'
    assert results[3]['value'] == '```\n```'
    assert results[4]['name'] == 'Heading 5'
    assert results[4]['value'] == '```\n```'
    a = Apprise()
    assert a.add('discord://{webhook_id}/{webhook_token}/?format=markdown&footer=Yes'.format(webhook_id=webhook_id, webhook_token=webhook_token)) is True
    NotifyDiscord.discord_max_fields = 1
    assert a.notify(body=test_markdown, title='title', notify_type=NotifyType.INFO, body_format=NotifyFormat.TEXT) is True
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    mock_post.side_effect = [response, response, response, requests.RequestException()]
    obj = Apprise.instantiate('discord://{}/{}/?format=markdown'.format(webhook_id, webhook_token))
    assert isinstance(obj, NotifyDiscord)
    assert obj.notify(body=test_markdown, title='title', notify_type=NotifyType.INFO) is False
    mock_post.side_effect = None
    (desc, results) = obj.extract_markdown_sections('')
    assert isinstance(results, list) is True
    assert len(results) == 0
    assert isinstance(desc, str) is True
    assert not desc
    test_markdown = 'Just a string without any header entries.\n' + 'A second line'
    (desc, results) = obj.extract_markdown_sections(test_markdown)
    assert isinstance(results, list) is True
    assert len(results) == 0
    assert isinstance(desc, str) is True
    assert desc == 'Just a string without any header entries.\n' + 'A second line'
    assert obj.notify(body=test_markdown, title='title', notify_type=NotifyType.INFO) is True
    a = Apprise()
    assert a.add('discord://{webhook_id}/{webhook_token}/?format=markdown&footer=Yes'.format(webhook_id=webhook_id, webhook_token=webhook_token)) is True
    assert a.notify(body=test_markdown, title='title', notify_type=NotifyType.INFO, body_format=NotifyFormat.TEXT) is True
    assert a.notify(body=test_markdown, title='title', notify_type=NotifyType.INFO, body_format=NotifyFormat.MARKDOWN) is True
    a.asset.image_url_logo = None
    assert a.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    a = Apprise()
    mock_post.reset_mock()
    assert a.add('discord://{webhook_id}/{webhook_token}/?thread=12345'.format(webhook_id=webhook_id, webhook_token=webhook_token)) is True
    assert a.notify(body='test', title='title') is True
    assert mock_post.call_count == 1
    response = mock_post.call_args_list[0][1]
    assert 'params' in response
    assert response['params'].get('thread_id') == '12345'

@mock.patch('requests.post')
def test_plugin_discord_markdown_extra(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyDiscord() Markdown Extra Checks\n\n    '
    webhook_id = 'A' * 24
    webhook_token = 'B' * 64
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    a = Apprise()
    assert a.add('discord://{webhook_id}/{webhook_token}/?format=markdown&footer=Yes'.format(webhook_id=webhook_id, webhook_token=webhook_token)) is True
    test_markdown = '[green-blue](https://google.com)'
    assert a.notify(body=test_markdown, title='title', notify_type=NotifyType.INFO, body_format=NotifyFormat.TEXT) is True
    assert a.notify(body='body', title='title', notify_type=NotifyType.INFO) is True

@mock.patch('requests.post')
def test_plugin_discord_attachments(mock_post):
    if False:
        return 10
    '\n    NotifyDiscord() Attachment Checks\n\n    '
    webhook_id = 'C' * 24
    webhook_token = 'D' * 64
    response = mock.Mock()
    response.status_code = requests.codes.ok
    bad_response = mock.Mock()
    bad_response.status_code = requests.codes.internal_server_error
    mock_post.return_value = response
    obj = Apprise.instantiate('discord://{}/{}/?format=markdown'.format(webhook_id, webhook_token))
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://discord.com/api/webhooks/{}/{}'.format(webhook_id, webhook_token)
    assert mock_post.call_args_list[1][0][0] == 'https://discord.com/api/webhooks/{}/{}'.format(webhook_id, webhook_token)
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    mock_post.return_value = None
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [side_effect]
        assert obj.send(body='test', attach=attach) is False
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [response, side_effect]
        assert obj.send(body='test', attach=attach) is False
    bad_response = mock.Mock()
    bad_response.status_code = requests.codes.internal_server_error
    mock_post.side_effect = [response, bad_response]
    assert obj.send(body='test', attach=attach) is False