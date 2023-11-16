import os
from unittest import mock
import requests
from json import dumps, loads
from datetime import datetime
from datetime import timezone
from apprise import Apprise
from apprise import NotifyType
from apprise import AppriseAttachment
from apprise.plugins.NotifyMastodon import NotifyMastodon
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('mastodon://', {'instance': None}), ('mastodon://:@/', {'instance': None}), ('mastodon://hostname', {'instance': TypeError}), ('toot://access_token@hostname', {'instance': NotifyMastodon}), ('toots://access_token@hostname', {'instance': NotifyMastodon, 'privacy_url': 'mastodons://****@hostname/'}), ('mastodon://access_token@hostname/@user/@user2', {'instance': NotifyMastodon, 'privacy_url': 'mastodon://****@hostname/@user/@user2'}), ('mastodon://hostname/@user/@user2?token=abcd123', {'instance': NotifyMastodon, 'privacy_url': 'mastodon://****@hostname/@user/@user2'}), ('mastodon://access_token@hostname?to=@user, @user2', {'instance': NotifyMastodon, 'privacy_url': 'mastodon://****@hostname/@user/@user2'}), ('mastodon://access_token@hostname/?cache=no', {'instance': NotifyMastodon}), ('mastodon://access_token@hostname/?spoiler=spoiler%20text', {'instance': NotifyMastodon}), ('mastodon://access_token@hostname/?language=en', {'instance': NotifyMastodon}), ('mastodons://access_token@hostname:8443', {'instance': NotifyMastodon}), ('mastodon://access_token@hostname/?key=My%20Idempotency%20Key', {'instance': NotifyMastodon}), ('mastodon://access_token@hostname/-/%/', {'instance': TypeError}), ('mastodon://access_token@hostname?visibility=invalid', {'instance': TypeError}), ('mastodon://access_token@hostname?visibility=direct', {'instance': NotifyMastodon, 'notify_response': False}), ('mastodon://access_token@hostname?visibility=direct', {'instance': NotifyMastodon, 'requests_response_text': {'id': '12345', 'username': 'test'}}), ('toots://access_token@hostname', {'instance': NotifyMastodon, 'test_requests_exceptions': True}))

def test_plugin_mastodon_urls():
    if False:
        return 10
    '\n    NotifyMastodon() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_mastodon_general(mock_post, mock_get):
    if False:
        i = 10
        return i + 15
    '\n    NotifyMastodon() General Tests\n\n    '
    token = 'access_key'
    host = 'nuxref.com'
    response_obj = {'username': 'caronc', 'id': 1234}
    epoch = datetime.fromtimestamp(0, timezone.utc)
    request = mock.Mock()
    request.content = dumps(response_obj)
    request.status_code = requests.codes.ok
    request.headers = {'X-RateLimit-Limit': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    mock_get.return_value = request
    mock_post.return_value = request
    obj = NotifyMastodon(token=token, host=host)
    assert isinstance(obj, NotifyMastodon) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    request.status_code = 403
    assert obj.send(body='test') is False
    assert obj.ratelimit_remaining == 1
    request.status_code = requests.codes.ok
    request.headers['X-RateLimit-Remaining'] = 0
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 0
    request.headers['X-RateLimit-Remaining'] = 10
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    del request.headers['X-RateLimit-Remaining']
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    request.headers['X-RateLimit-Remaining'] = 1
    del request.headers['X-RateLimit-Limit']
    assert obj.send(body='test') is True
    request.headers['X-RateLimit-Limit'] = (datetime.now(timezone.utc) - epoch).total_seconds() + 1
    request.headers['X-RateLimit-Remaining'] = 0
    obj.ratelimit_remaining = 0
    assert obj.send(body='test') is True
    request.headers['X-RateLimit-Limit'] = (datetime.now(timezone.utc) - epoch).total_seconds() - 1
    request.headers['X-RateLimit-Remaining'] = 0
    obj.ratelimit_remaining = 0
    assert obj.send(body='test') is True
    request.headers['X-RateLimit-Limit'] = (datetime.now(timezone.utc) - epoch).total_seconds()
    request.headers['X-RateLimit-Remaining'] = 1
    obj.ratelimit_remaining = 1
    obj.targets.append('usera')
    request.content = dumps(response_obj)
    response_obj = {'username': 'usera', 'id': 4321}
    request.content = None
    assert obj.send(body='test') is True
    request.content = '{'
    assert obj.send(body='test') is True
    request.content = '{}'
    results = NotifyMastodon.parse_url('mastodon://{}@{}/@user?visbility=direct'.format(token, host))
    assert isinstance(results, dict) is True
    assert '@user' in results['targets']
    response_obj = None
    assert obj.send(body='test') is True
    response_obj = '{'
    assert obj.send(body='test') is True
    mock_get.reset_mock()
    mock_post.reset_mock()
    request = mock.Mock()
    request.content = dumps({'id': '1234', 'username': 'caronc'})
    request.status_code = requests.codes.ok
    mock_get.return_value = request
    mastodon_url = 'mastodons://key@host?visibility=direct'
    obj = Apprise.instantiate(mastodon_url)
    obj._whoami(lazy=True)
    assert mock_get.call_count == 1
    assert mock_get.call_args_list[0][0][0] == 'https://host/api/v1/accounts/verify_credentials'
    mock_get.reset_mock()
    obj._whoami(lazy=True)
    assert mock_get.call_count == 0
    mock_get.reset_mock()
    obj._whoami(lazy=False)
    assert mock_get.call_count == 1
    assert mock_get.call_args_list[0][0][0] == 'https://host/api/v1/accounts/verify_credentials'

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_mastodon_attachments(mock_get, mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyMastodon() Toot Attachment Checks\n\n    '
    akey = 'access_key'
    host = 'nuxref.com'
    username = 'caronc'
    good_response_obj = {'id': '1234'}
    good_response = mock.Mock()
    good_response.content = dumps(good_response_obj)
    good_response.status_code = requests.codes.ok
    good_whoami_response_obj = {'username': username, 'id': '9876'}
    good_whoami_response = mock.Mock()
    good_whoami_response.content = dumps(good_whoami_response_obj)
    good_whoami_response.status_code = requests.codes.ok
    bad_response = mock.Mock()
    bad_response.content = dumps({})
    bad_response.status_code = requests.codes.internal_server_error
    good_media_response = mock.Mock()
    good_media_response.content = dumps({'id': '710511363345354753', 'file_mime': 'image/jpeg'})
    good_media_response.status_code = requests.codes.ok
    mock_post.side_effect = [good_media_response, good_response]
    mock_get.return_value = good_whoami_response
    mastodon_url = 'mastodon://{}@{}'.format(akey, host)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    obj = Apprise.instantiate(mastodon_url)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_get.call_count == 0
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'http://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[1][0][0] == 'http://nuxref.com/api/v1/statuses'
    assert 'files' in mock_post.call_args_list[0][1]
    assert 'file' in mock_post.call_args_list[0][1]['files']
    assert mock_post.call_args_list[0][1]['files']['file'][0] == 'apprise-test.gif'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert 'status' in payload
    assert payload['status'] == 'title\r\nbody'
    assert 'sensitive' in payload
    assert payload['sensitive'] is False
    assert 'media_ids' in payload
    assert isinstance(payload['media_ids'], list)
    assert len(payload['media_ids']) == 1
    assert payload['media_ids'][0] == '710511363345354753'
    assert 'spoiler_text' not in payload
    mock_get.reset_mock()
    mock_post.reset_mock()
    mock_post.side_effect = [good_media_response, good_response]
    mock_get.return_value = good_whoami_response
    mastodon_url = 'mastodon://{}@{}?visibility=direct'.format(akey, host)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    obj = Apprise.instantiate(mastodon_url)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_get.call_count == 1
    assert mock_post.call_count == 2
    assert mock_get.call_args_list[0][0][0] == 'http://nuxref.com/api/v1/accounts/verify_credentials'
    assert mock_post.call_args_list[0][0][0] == 'http://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[1][0][0] == 'http://nuxref.com/api/v1/statuses'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert 'status' in payload
    assert payload['status'] == '@caronc title\r\nbody'
    assert 'sensitive' in payload
    assert payload['sensitive'] is False
    assert 'media_ids' in payload
    assert isinstance(payload['media_ids'], list)
    assert len(payload['media_ids']) == 1
    assert payload['media_ids'][0] == '710511363345354753'
    mock_get.reset_mock()
    mock_post.reset_mock()
    attach = (AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif')), AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.png')), AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.jpeg')))
    mr1 = mock.Mock()
    mr1.content = dumps({'id': '1', 'file_mime': 'image/gif'})
    mr1.status_code = requests.codes.ok
    mr2 = mock.Mock()
    mr2.content = dumps({'id': '2', 'file_mime': 'image/png'})
    mr2.status_code = requests.codes.ok
    mr3 = mock.Mock()
    mr3.content = dumps({'id': '3', 'file_mime': 'image/jpeg'})
    mr3.status_code = requests.codes.ok
    mock_post.side_effect = [mr1, mr2, mr3, good_response, good_response]
    mock_get.return_value = good_whoami_response
    mastodon_url = 'mastodons://{}@{}?visibility=direct&sensitive=yes&key=abcd'.format(akey, host)
    obj = Apprise.instantiate(mastodon_url)
    assert obj.notify(body='Check this out @caronc', title='Apprise', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_get.call_count == 1
    assert mock_post.call_count == 5
    assert mock_post.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[1][0][0] == 'https://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[2][0][0] == 'https://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[3][0][0] == 'https://nuxref.com/api/v1/statuses'
    assert mock_post.call_args_list[4][0][0] == 'https://nuxref.com/api/v1/statuses'
    assert mock_get.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/accounts/verify_credentials'
    payload = loads(mock_post.call_args_list[3][1]['data'])
    assert 'status' in payload
    assert payload['status'] == 'Apprise\r\nCheck this out @caronc'
    assert 'sensitive' in payload
    assert payload['sensitive'] is True
    assert 'language' not in payload
    assert 'Idempotency-Key' in payload
    assert payload['Idempotency-Key'] == 'abcd'
    assert 'media_ids' in payload
    assert isinstance(payload['media_ids'], list)
    assert len(payload['media_ids']) == 1
    assert payload['media_ids'][0] == '1'
    payload = loads(mock_post.call_args_list[4][1]['data'])
    assert 'status' in payload
    assert payload['status'] == '02/02'
    assert 'sensitive' in payload
    assert payload['sensitive'] is False
    assert 'language' not in payload
    assert 'Idempotency-Key' in payload
    assert payload['Idempotency-Key'] == 'abcd-part01'
    assert 'media_ids' in payload
    assert isinstance(payload['media_ids'], list)
    assert len(payload['media_ids']) == 2
    assert '2' in payload['media_ids']
    assert '3' in payload['media_ids']
    mock_get.reset_mock()
    mock_post.reset_mock()
    mock_post.side_effect = [mr1, mr2, mr3, good_response, good_response]
    mock_get.return_value = good_whoami_response
    assert obj.notify(body='Check this out @caronc', title='Apprise', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 5
    assert mock_get.call_count == 0
    mock_get.reset_mock()
    mock_post.reset_mock()
    attach = (os.path.join(TEST_VAR_DIR, 'apprise-test.png'), os.path.join(TEST_VAR_DIR, 'apprise-test.jpeg'))
    mock_post.side_effect = [mr2, mr3, good_response, good_response]
    mock_get.return_value = good_whoami_response
    mastodon_url = 'mastodons://{}@{}?batch=no'.format(akey, host)
    obj = Apprise.instantiate(mastodon_url)
    assert obj.notify(body='Check this out @caronc', title='Apprise', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 4
    assert mock_get.call_count == 0
    assert mock_post.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[1][0][0] == 'https://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[2][0][0] == 'https://nuxref.com/api/v1/statuses'
    assert mock_post.call_args_list[3][0][0] == 'https://nuxref.com/api/v1/statuses'
    mock_get.reset_mock()
    mock_post.reset_mock()
    bad_response = mock.Mock()
    bad_response.status_code = requests.codes.internal_server_error
    bad_responses = (dumps({'error': 'authorized scopes'}), '')
    for response in bad_responses:
        mock_post.side_effect = [good_media_response, bad_response]
        bad_response.content = response
        mastodon_url = 'mastodons://{}@{}?visibility=public&spoiler=uhoh'.format(akey, host)
        obj = Apprise.instantiate(mastodon_url)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
        assert mock_get.call_count == 0
        assert mock_post.call_count == 2
        assert mock_post.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/media'
        assert mock_post.call_args_list[1][0][0] == 'https://nuxref.com/api/v1/media'
        mock_get.reset_mock()
        mock_post.reset_mock()
    for response in bad_responses:
        mock_post.side_effect = [bad_response]
        bad_response.content = response
        mastodon_url = 'mastodons://{}@{}'.format(akey, host)
        obj = Apprise.instantiate(mastodon_url)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
        assert mock_get.call_count == 0
        assert mock_post.call_count == 1
        assert mock_post.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/statuses'
        mock_get.reset_mock()
        mock_post.reset_mock()
    for response in bad_responses:
        mock_get.side_effect = [bad_response]
        bad_response.content = response
        mastodon_url = 'mastodons://{}@{}?visibility=direct'.format(akey, host)
        obj = Apprise.instantiate(mastodon_url)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
        assert mock_get.call_count == 1
        assert mock_post.call_count == 0
        assert mock_get.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/accounts/verify_credentials'
        mock_get.reset_mock()
        mock_post.reset_mock()
    mock_post.side_effect = [mr1, mr2, mr3, good_response, good_response]
    mock_get.return_value = None
    mastodon_url = 'mastodons://{}@{}'.format(akey, host)
    obj = Apprise.instantiate(mastodon_url)
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    assert mock_get.call_count == 0
    assert mock_post.call_count == 0
    mock_get.reset_mock()
    mock_post.reset_mock()
    mock_post.side_effect = [good_media_response, OSError(), good_media_response]
    mock_get.return_value = good_response
    attach = [os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-archive.zip'), os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')]
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_get.call_count == 0
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://nuxref.com/api/v1/media'
    assert mock_post.call_args_list[1][0][0] == 'https://nuxref.com/api/v1/media'