"""
Created on Sun Oct 04 23:10:41 2015
@author: ujjwal.karn
"""
import tweepy
from tweepy import OAuthHandler
access_token = 'xxxxxxxx'
access_token_secret = 'xxxxxxxx'
consumer_key = 'xxxxxxxx'
consumer_secret = 'xxxxxxxx'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
from tweepy import Stream
from tweepy.streaming import StreamListener

class MyListener(StreamListener):

    def on_data(self, data):
        if False:
            return 10
        try:
            with open('location/file_name.txt', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print('Error on_data: %s' % str(e))
        return True

    def on_error(self, status):
        if False:
            print('Hello World!')
        print(status)
        return True
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#cricket'])