import json
import logging
import os
from datetime import datetime
from datetime import timezone
from unittest.mock import Mock, patch
import pytest
import requests
from apprise import Apprise
from apprise import NotifyType
from apprise import AppriseAttachment
from apprise.plugins.NotifyTwitter import NotifyTwitter
from helpers import AppriseURLTester
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
TWITTER_SCREEN_NAME = 'apprise'
apprise_url_tests = (('twitter://', {'instance': TypeError}), ('twitter://:@/', {'instance': TypeError}), ('twitter://consumer_key', {'instance': TypeError}), ('twitter://consumer_key/consumer_secret/', {'instance': TypeError}), ('twitter://consumer_key/consumer_secret/atoken1/', {'instance': TypeError}), ('twitter://consumer_key/consumer_secret/atoken2/access_secret', {'instance': NotifyTwitter, 'notify_response': False, 'privacy_url': 'x://c...y/****/a...2/****'}), ('twitter://consumer_key/consumer_secret/atoken3/access_secret?cache=no', {'instance': NotifyTwitter, 'requests_response_text': {'id': 12345, 'screen_name': 'test', 'media_id': 123}}), ('twitter://consumer_key/consumer_secret/atoken4/access_secret', {'instance': NotifyTwitter, 'requests_response_text': {'id': 12345, 'screen_name': 'test', 'media_id': 123}}), ('twitter://consumer_key/consumer_secret/atoken5/access_secret', {'instance': NotifyTwitter, 'requests_response_text': {'id': 12345, 'screen_name': 'test', 'media_id': 123}}), ('twitter://consumer_key/consumer_secret2/atoken6/access_secret', {'instance': NotifyTwitter, 'requests_response_text': {'id': 12345, 'media_id': 123}, 'notify_response': False}), ('twitter://user@consumer_key/csecret2/atoken7/access_secret/-/%/', {'instance': NotifyTwitter, 'notify_response': False}), ('twitter://user@consumer_key/csecret/atoken8/access_secret?cache=No&batch=No', {'instance': NotifyTwitter, 'requests_response_text': [{'id': 12345, 'screen_name': 'user'}]}), ('twitter://user@consumer_key/csecret/atoken9/access_secret', {'instance': NotifyTwitter, 'requests_response_text': [{'id': 12345, 'screen_name': 'user'}]}), ('twitter://user@consumer_key/csecret/atoken11/access_secret', {'instance': NotifyTwitter, 'notify_response': False}), ('tweet://ckey/csecret/atoken12/access_secret', {'instance': NotifyTwitter}), ('twitter://user@ckey/csecret/atoken13/access_secret?mode=invalid', {'instance': TypeError}), ('twitter://usera@consumer_key/consumer_secret/atoken14/access_secret/user/?to=userb', {'instance': NotifyTwitter, 'requests_response_text': [{'id': 12345, 'screen_name': 'usera'}, {'id': 12346, 'screen_name': 'userb'}, {'id': 123}]}), ('twitter://ckey/csecret/atoken15/access_secret', {'instance': NotifyTwitter, 'response': False, 'requests_response_code': 999}), ('twitter://ckey/csecret/atoken16/access_secret', {'instance': NotifyTwitter, 'test_requests_exceptions': True}), ('twitter://ckey/csecret/atoken17/access_secret?mode=tweet', {'instance': NotifyTwitter, 'test_requests_exceptions': True}))

def good_response(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare a good response.\n    '
    response = Mock()
    response.content = json.dumps(data)
    response.status_code = requests.codes.ok
    return response

def bad_response(data):
    if False:
        return 10
    '\n    Prepare a bad response.\n    '
    response = Mock()
    response.content = json.dumps(data)
    response.status_code = requests.codes.internal_server_error
    return response

@pytest.fixture
def twitter_url():
    if False:
        while True:
            i = 10
    ckey = 'ckey'
    csecret = 'csecret'
    akey = 'akey'
    asecret = 'asecret'
    url = 'twitter://{}/{}/{}/{}'.format(ckey, csecret, akey, asecret)
    return url

@pytest.fixture
def good_message_response():
    if False:
        i = 10
        return i + 15
    '\n    Prepare a good tweet response.\n    '
    response = good_response({'screen_name': TWITTER_SCREEN_NAME, 'id': 9876})
    return response

@pytest.fixture
def bad_message_response():
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare a bad message response.\n    '
    response = bad_response({'errors': [{'code': 999, 'message': 'Something failed'}]})
    return response

@pytest.fixture
def good_media_response():
    if False:
        print('Hello World!')
    '\n    Prepare a good media response.\n    '
    response = Mock()
    response.content = json.dumps({'media_id': 710511363345354753, 'media_id_string': '710511363345354753', 'media_key': '3_710511363345354753', 'size': 11065, 'expires_after_secs': 86400, 'image': {'image_type': 'image/jpeg', 'w': 800, 'h': 320}})
    response.status_code = requests.codes.ok
    return response

@pytest.fixture
def bad_media_response():
    if False:
        print('Hello World!')
    '\n    Prepare a bad media response.\n    '
    response = bad_response({'errors': [{'code': 93, 'message': 'This application is not allowed to access or delete your direct messages.'}]})
    return response

@pytest.fixture(autouse=True)
def ensure_get_verify_credentials_is_mocked(mocker, good_message_response):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make sure requests to https://api.twitter.com/1.1/account/verify_credentials.json\n    do not escape the test harness, for all test case functions.\n    '
    mock_get = mocker.patch('requests.get')
    mock_get.return_value = good_message_response

def test_plugin_twitter_urls():
    if False:
        return 10
    '\n    NotifyTwitter() Apprise URLs\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_twitter_general(mocker):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTwitter() General Tests\n    '
    mock_get = mocker.patch('requests.get')
    mock_post = mocker.patch('requests.post')
    ckey = 'ckey'
    csecret = 'csecret'
    akey = 'akey'
    asecret = 'asecret'
    response_obj = [{'screen_name': TWITTER_SCREEN_NAME, 'id': 9876}]
    epoch = datetime.fromtimestamp(0, timezone.utc)
    request = Mock()
    request.content = json.dumps(response_obj)
    request.status_code = requests.codes.ok
    request.headers = {'x-rate-limit-reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'x-rate-limit-remaining': 1}
    mock_get.return_value = request
    mock_post.return_value = request
    obj = NotifyTwitter(ckey=ckey, csecret=csecret, akey=akey, asecret=asecret, targets=TWITTER_SCREEN_NAME)
    assert isinstance(obj, NotifyTwitter) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    request.status_code = 403
    assert obj.send(body='test') is False
    assert obj.ratelimit_remaining == 1
    request.status_code = requests.codes.ok
    request.headers['x-rate-limit-remaining'] = 0
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 0
    request.headers['x-rate-limit-remaining'] = 10
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    del request.headers['x-rate-limit-remaining']
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    request.headers['x-rate-limit-remaining'] = 1
    del request.headers['x-rate-limit-reset']
    assert obj.send(body='test') is True
    request.headers['x-rate-limit-reset'] = (datetime.now(timezone.utc) - epoch).total_seconds() + 1
    request.headers['x-rate-limit-remaining'] = 0
    obj.ratelimit_remaining = 0
    assert obj.send(body='test') is True
    request.headers['x-rate-limit-reset'] = (datetime.now(timezone.utc) - epoch).total_seconds() - 1
    request.headers['x-rate-limit-remaining'] = 0
    obj.ratelimit_remaining = 0
    assert obj.send(body='test') is True
    request.headers['x-rate-limit-reset'] = (datetime.now(timezone.utc) - epoch).total_seconds()
    request.headers['x-rate-limit-remaining'] = 1
    obj.ratelimit_remaining = 1
    obj.targets.append('usera')
    request.content = json.dumps(response_obj)
    response_obj = [{'screen_name': 'usera', 'id': 1234}]
    assert obj.send(body='test') is True
    NotifyTwitter._user_cache = {}
    assert obj.send(body='test') is True
    request.content = None
    assert obj.send(body='test') is True
    request.content = '{'
    assert obj.send(body='test') is True
    request.content = '{}'
    results = NotifyTwitter.parse_url('twitter://{}/{}/{}/{}?to={}'.format(ckey, csecret, akey, asecret, TWITTER_SCREEN_NAME))
    assert isinstance(results, dict) is True
    assert TWITTER_SCREEN_NAME in results['targets']
    response_obj = None
    assert obj.send(body='test') is True
    response_obj = '{'
    assert obj.send(body='test') is True
    NotifyTwitter._user_cache = {}
    response_obj = {'screen_name': TWITTER_SCREEN_NAME, 'id': 9876}
    request.content = json.dumps(response_obj)
    obj = NotifyTwitter(ckey=ckey, csecret=csecret, akey=akey, asecret=asecret)
    assert obj.send(body='test') is True
    NotifyTwitter._user_cache = {}
    NotifyTwitter._whoami_cache = None
    obj.ckey = 'different.then.it.was'
    assert obj.send(body='test') is True
    NotifyTwitter._whoami_cache = None
    obj.ckey = 'different.again'
    assert obj.send(body='test') is True

def test_plugin_twitter_edge_cases():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTwitter() Edge Cases\n    '
    with pytest.raises(TypeError):
        NotifyTwitter(ckey=None, csecret=None, akey=None, asecret=None)
    with pytest.raises(TypeError):
        NotifyTwitter(ckey='value', csecret=None, akey=None, asecret=None)
    with pytest.raises(TypeError):
        NotifyTwitter(ckey='value', csecret='value', akey=None, asecret=None)
    with pytest.raises(TypeError):
        NotifyTwitter(ckey='value', csecret='value', akey='value', asecret=None)
    assert isinstance(NotifyTwitter(ckey='value', csecret='value', akey='value', asecret='value'), NotifyTwitter)
    assert isinstance(NotifyTwitter(ckey='value', csecret='value', akey='value', asecret='value', user='l2gnux'), NotifyTwitter)
    with pytest.raises(TypeError):
        NotifyTwitter(ckey='value', csecret='value', akey='value', asecret='value', targets='%G@rB@g3')

def test_plugin_twitter_dm_caching(mocker, twitter_url, good_message_response, good_media_response):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that the `NotifyTwitter.{_user_cache,_whoami_cache}` caches\n    work as intended.\n    '
    mock_get = mocker.patch('requests.get')
    mock_get.return_value = good_message_response
    mock_post = mocker.patch('requests.post')
    mock_post.side_effect = [good_message_response, good_message_response]
    if hasattr(NotifyTwitter, '_user_cache'):
        NotifyTwitter._user_cache = {}
    if hasattr(NotifyTwitter, '_whoami_cache'):
        NotifyTwitter._whoami_cache = {}
    obj = Apprise.instantiate(twitter_url)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_get.call_count == 1
    assert mock_get.call_args_list[0][0][0] == 'https://api.twitter.com/1.1/account/verify_credentials.json'
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'
    mock_get.reset_mock()
    mock_post.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_get.call_count == 0
    assert mock_post.call_count == 1

def test_plugin_twitter_dm_attachments_basic(mocker, twitter_url, good_message_response, good_media_response):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTwitter() DM Attachment Checks - Basic\n    '
    mock_get = mocker.patch('requests.get')
    mock_post = mocker.patch('requests.post')
    epoch = datetime.fromtimestamp(0, timezone.utc)
    mock_get.return_value = good_message_response
    mock_post.return_value.headers = {'x-rate-limit-reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'x-rate-limit-remaining': 1}
    mock_post.side_effect = [good_media_response, good_message_response]
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_get.call_count == 1
    assert mock_get.call_args_list[0][0][0] == 'https://api.twitter.com/1.1/account/verify_credentials.json'
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'

def test_plugin_twitter_dm_attachments_message_fails(mocker, twitter_url, good_media_response, bad_message_response):
    if False:
        return 10
    '\n    Test case with a bad media response.\n    '
    mock_post = mocker.patch('requests.post')
    mock_post.side_effect = [good_media_response, bad_message_response]
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'

def test_plugin_twitter_dm_attachments_upload_fails(mocker, twitter_url, good_message_response, bad_media_response):
    if False:
        return 10
    '\n    Test case where upload fails.\n    '
    mock_post = mocker.patch('requests.post')
    mock_post.side_effect = [bad_media_response, good_message_response]
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'

def test_plugin_twitter_dm_attachments_invalid_attachment(mocker, twitter_url, good_message_response):
    if False:
        return 10
    '\n    Test case with an invalid attachment.\n    '
    mock_post: Mock = mocker.patch('requests.post')
    mock_post.side_effect = [good_media_response, good_message_response]
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.mock_calls == []

def test_plugin_twitter_dm_attachments_multiple(mocker, twitter_url, good_message_response, good_media_response):
    if False:
        for i in range(10):
            print('nop')
    mock_post = mocker.patch('requests.post')
    mock_post.side_effect = [good_media_response, good_media_response, good_media_response, good_media_response, good_message_response, good_message_response, good_message_response, good_message_response]
    attach = [os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.jpeg'), os.path.join(TEST_VAR_DIR, 'apprise-test.png'), os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')]
    obj = Apprise.instantiate(twitter_url)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 8
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[2][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[3][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[4][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'
    assert mock_post.call_args_list[5][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'
    assert mock_post.call_args_list[6][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'
    assert mock_post.call_args_list[7][0][0] == 'https://api.twitter.com/1.1/direct_messages/events/new.json'

def test_plugin_twitter_dm_attachments_multiple_oserror(mocker, twitter_url, good_message_response, good_media_response):
    if False:
        for i in range(10):
            print('nop')
    mock_post = mocker.patch('requests.post')
    mock_post.side_effect = [good_media_response, OSError()]
    attach = [os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.png'), os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')]
    obj = Apprise.instantiate(twitter_url)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_basic(mock_post, twitter_url, good_message_response, good_media_response):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTwitter() Tweet Attachment Checks - Basic\n    '
    mock_post.side_effect = [good_media_response, good_message_response]
    twitter_url += '?mode=tweet'
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_more_logging(mock_post, twitter_url, good_media_response):
    if False:
        print('Hello World!')
    '\n    NotifyTwitter() Tweet Attachment Checks - More logging\n\n    TODO: The "more logging" aspect is not verified yet?\n    '
    good_tweet_response = good_response({'screen_name': TWITTER_SCREEN_NAME, 'id': 9876, 'id_str': '12345', 'user': {'screen_name': TWITTER_SCREEN_NAME}})
    mock_post.side_effect = [good_media_response, good_tweet_response]
    twitter_url += '?mode=tweet'
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_bad_message_response(mock_post, twitter_url, good_media_response, bad_message_response):
    if False:
        i = 10
        return i + 15
    mock_post.side_effect = [good_media_response, bad_message_response]
    twitter_url += '?mode=tweet'
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_bad_message_response_unparseable(mock_post, twitter_url, good_media_response):
    if False:
        print('Hello World!')
    bad_message_response = bad_response('')
    mock_post.side_effect = [good_media_response, bad_message_response]
    twitter_url += '?mode=tweet'
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_upload_fails(mock_post, twitter_url, good_media_response):
    if False:
        return 10
    bad_tweet_response = bad_response({})
    mock_post.side_effect = [good_media_response, bad_tweet_response]
    twitter_url += '?mode=tweet'
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_invalid_attachment(mock_post, twitter_url, good_message_response, good_media_response):
    if False:
        while True:
            i = 10
    mock_post.side_effect = [good_media_response, good_message_response]
    twitter_url += '?mode=tweet'
    obj = Apprise.instantiate(twitter_url)
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 0

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_multiple_batch(mock_post, twitter_url, good_message_response, good_media_response):
    if False:
        return 10
    mock_post.side_effect = [good_media_response, good_media_response, good_media_response, good_media_response, good_message_response, good_message_response, good_message_response, good_message_response]
    obj = Apprise.instantiate(twitter_url + '?mode=tweet')
    attach = [os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.jpeg'), os.path.join(TEST_VAR_DIR, 'apprise-test.png'), os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')]
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 7
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[2][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[3][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[4][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'
    assert mock_post.call_args_list[5][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'
    assert mock_post.call_args_list[6][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_multiple_nobatch(mock_post, twitter_url, good_message_response, good_media_response):
    if False:
        while True:
            i = 10
    mock_post.side_effect = [good_media_response, good_media_response, good_media_response, good_media_response, good_message_response, good_message_response, good_message_response, good_message_response]
    obj = Apprise.instantiate(twitter_url + '?mode=tweet&batch=no')
    attach = [os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.jpeg'), os.path.join(TEST_VAR_DIR, 'apprise-test.png'), os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')]
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 8
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[2][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[3][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[4][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'
    assert mock_post.call_args_list[5][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'
    assert mock_post.call_args_list[6][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'
    assert mock_post.call_args_list[7][0][0] == 'https://api.twitter.com/1.1/statuses/update.json'

@patch('requests.post')
def test_plugin_twitter_tweet_attachments_multiple_oserror(mock_post, twitter_url, good_media_response):
    if False:
        for i in range(10):
            print('nop')
    mock_post.side_effect = [good_media_response, OSError()]
    attach = [os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.png'), os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')]
    obj = Apprise.instantiate(twitter_url + '?mode=tweet')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'
    assert mock_post.call_args_list[1][0][0] == 'https://upload.twitter.com/1.1/media/upload.json'