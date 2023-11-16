import requests
from apprise.plugins.NotifyReddit import NotifyReddit
from helpers import AppriseURLTester
from unittest import mock
from json import dumps
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('reddit://', {'instance': TypeError}), ('reddit://:@/', {'instance': TypeError}), ('reddit://user@app_id/app_secret/', {'instance': TypeError}), ('reddit://user:password@app_id/', {'instance': TypeError}), ('reddit://user:password@app%id/appsecret/apprise', {'instance': TypeError}), ('reddit://user:password@app%id/app_secret/apprise', {'instance': TypeError}), ('reddit://user:password@app-id/app-secret/apprise?kind=invalid', {'instance': TypeError}), ('reddit://user:password@app-id/app-secret/apprise', {'instance': NotifyReddit, 'notify_response': False}), ('reddit://user:password@app-id/app-secret', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}}, 'notify_response': False}), ('reddit://user:password@app-id/app-secret/apprise', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': [('KEY', 'DESC', 'INFO')]}}, 'notify_response': False}), ('reddit://user:password@app-id/app-secret/apprise', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}}, 'privacy_url': 'reddit://user:****@****/****/apprise'}), ('reddit://user:password@app-id/app-secret/apprise/subreddit2', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}}, 'privacy_url': 'reddit://user:****@****/****/apprise/subreddit2'}), ('reddit://user:pass@id/secret/sub/?ad=yes&nsfw=yes&replies=no&resubmit=yes&spoiler=yes&kind=self', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}}, 'privacy_url': 'reddit://user:****@****/****/sub'}), ('reddit://?user=l2g&pass=pass&app_secret=abc123&app_id=54321&to=sub1,sub2', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}}, 'privacy_url': 'reddit://l2g:****@****/****/sub1/sub2'}), ('reddit://user:pass@id/secret/sub7/sub6/sub5/?flair_id=wonder&flair_text=not%20for%20you', {'instance': NotifyReddit, 'requests_response_text': {'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}}, 'privacy_url': 'reddit://user:****@****/****/sub'}), ('reddit://user:password@app-id/app-secret/apprise', {'instance': NotifyReddit, 'response': False, 'requests_response_code': 999}), ('reddit://user:password@app-id/app-secret/apprise', {'instance': NotifyReddit, 'test_requests_exceptions': True}))

def test_plugin_reddit_urls():
    if False:
        return 10
    '\n    NotifyReddit() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_reddit_general(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyReddit() General Tests\n\n    '
    NotifyReddit.clock_skew = timedelta(seconds=0)
    kwargs = {'app_id': 'a' * 10, 'app_secret': 'b' * 20, 'user': 'user', 'password': 'pasword', 'targets': 'apprise'}
    epoch = datetime.fromtimestamp(0, timezone.utc)
    good_response = mock.Mock()
    good_response.content = dumps({'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}})
    good_response.status_code = requests.codes.ok
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    mock_post.return_value = good_response
    obj = NotifyReddit(**kwargs)
    assert isinstance(obj, NotifyReddit) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='http://hostname') is True
    bad_response = mock.Mock()
    bad_response.content = ''
    bad_response.status_code = 401
    mock_post.return_value = bad_response
    assert obj.send(body='test') is False
    assert obj.ratelimit_remaining == 1
    mock_post.return_value = good_response
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 0}
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 0
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 10}
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    del good_response.headers['X-RateLimit-Remaining']
    assert obj.send(body='test') is True
    assert obj.ratelimit_remaining == 10
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    del good_response.headers['X-RateLimit-Reset']
    assert obj.send(body='test') is True
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds() + 1, 'X-RateLimit-Remaining': 0}
    obj.ratelimit_remaining = 0
    assert obj.send(body='test') is True
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds() - 1, 'X-RateLimit-Remaining': 0}
    assert obj.send(body='test') is True
    obj.ratelimit_remaining = 1
    response = mock.Mock()
    response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    response.content = '{'
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is False
    response.content = '{}'
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is False
    response.content = dumps({'access_token': '', 'json': {'errors': []}})
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is False
    response.content = None
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is False
    good_response.content = dumps({'access_token': 'abc123', 'token_type': 'bearer', 'expires_in': 100000, 'scope': '*', 'refresh_token': 'def456', 'json': {'errors': []}})
    good_response.status_code = requests.codes.ok
    good_response.headers = {'X-RateLimit-Reset': (datetime.now(timezone.utc) - epoch).total_seconds(), 'X-RateLimit-Remaining': 1}
    mock_post.reset_mock()
    mock_post.side_effect = [good_response, bad_response, good_response, good_response]
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is True
    assert mock_post.call_count == 4
    assert mock_post.call_args_list[0][0][0] == 'https://www.reddit.com/api/v1/access_token'
    assert mock_post.call_args_list[1][0][0] == 'https://oauth.reddit.com/api/submit'
    assert mock_post.call_args_list[2][0][0] == 'https://www.reddit.com/api/v1/access_token'
    assert mock_post.call_args_list[3][0][0] == 'https://oauth.reddit.com/api/submit'
    mock_post.side_effect = [good_response, bad_response, bad_response]
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is False
    response.content = '{'
    response.status_code = requests.codes.ok
    mock_post.side_effect = [good_response, bad_response, good_response, response]
    obj = NotifyReddit(**kwargs)
    assert obj.send(body='test') is False