import unittest
from unittest.mock import MagicMock, patch
from superagi.tools.twitter.send_tweets import SendTweetsInput, SendTweetsTool

class TestSendTweetsInput(unittest.TestCase):

    def test_fields(self):
        if False:
            for i in range(10):
                print('nop')
        data = SendTweetsInput(tweet_text='Hello world', is_media=True, media_files=['image1.png', 'image2.png'])
        self.assertEqual(data.tweet_text, 'Hello world')
        self.assertEqual(data.is_media, True)
        self.assertEqual(data.media_files, ['image1.png', 'image2.png'])

class TestSendTweetsTool(unittest.TestCase):

    @patch('superagi.helper.twitter_tokens.TwitterTokens.get_twitter_creds', return_value={'token': '123', 'token_secret': '456'})
    @patch('superagi.helper.twitter_helper.TwitterHelper.get_media_ids', return_value=[789])
    @patch('superagi.helper.twitter_helper.TwitterHelper.send_tweets')
    def test_execute(self, mock_send_tweets, mock_get_media_ids, mock_get_twitter_creds):
        if False:
            print('Hello World!')
        responseMock = MagicMock()
        responseMock.status_code = 201
        mock_send_tweets.return_value = responseMock
        obj = SendTweetsTool()
        obj.toolkit_config = MagicMock()
        obj.toolkit_config.toolkit_id = 1
        obj.toolkit_config.session = MagicMock()
        obj.agent_id = 99
        obj.agent_execution_id = 1
        self.assertEqual(obj._execute(True), 'Tweet posted successfully!!')
        mock_get_twitter_creds.assert_called_once_with(1)
        mock_send_tweets.assert_called_once_with({'media': {'media_ids': [789]}, 'text': 'None'}, {'token': '123', 'token_secret': '456'})
        mock_get_twitter_creds.reset_mock()
        mock_get_media_ids.reset_mock()
        mock_send_tweets.reset_mock()
        responseMock.status_code = 400
        self.assertEqual(obj._execute(False, 'Hello world', ['image1.png']), 'Error posting tweet. (Status code: 400)')
        mock_get_twitter_creds.assert_called_once_with(1)
        mock_get_media_ids.assert_not_called()
        mock_send_tweets.assert_called_once_with({'text': 'Hello world'}, {'token': '123', 'token_secret': '456'})
if __name__ == '__main__':
    unittest.main()