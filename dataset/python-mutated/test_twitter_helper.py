import unittest
from unittest.mock import Mock, patch
from requests.models import Response
from requests_oauthlib import OAuth1Session
from superagi.helper.twitter_helper import TwitterHelper

class TestSendTweets(unittest.TestCase):

    @patch.object(OAuth1Session, 'post')
    def test_send_tweets_success(self, mock_post):
        if False:
            while True:
                i = 10
        test_params = {'status': 'Hello, Twitter!'}
        test_creds = Mock()
        test_oauth = OAuth1Session(test_creds.api_key)
        resp = Response()
        resp.status_code = 200
        mock_post.return_value = resp
        response = TwitterHelper().send_tweets(test_params, test_creds)
        test_oauth.post.assert_called_once_with('https://api.twitter.com/2/tweets', json=test_params)
        self.assertEqual(response.status_code, 200)

    @patch.object(OAuth1Session, 'post')
    def test_send_tweets_failure(self, mock_post):
        if False:
            i = 10
            return i + 15
        test_params = {'status': 'Hello, Twitter!'}
        test_creds = Mock()
        test_oauth = OAuth1Session(test_creds.api_key)
        resp = Response()
        resp.status_code = 400
        mock_post.return_value = resp
        response = TwitterHelper().send_tweets(test_params, test_creds)
        test_oauth.post.assert_called_once_with('https://api.twitter.com/2/tweets', json=test_params)
        self.assertEqual(response.status_code, 400)
if __name__ == '__main__':
    unittest.main()