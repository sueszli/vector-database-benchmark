from __future__ import unicode_literals
from __future__ import print_function
import pickle

class MockFeedParser(object):

    def parse(self, url):
        if False:
            i = 10
            return i + 15
        if url == 'user_feed':
            with open('tests/data/user_feed.p', 'rb') as fp:
                return pickle.load(fp)
        else:
            with open('tests/data/trends.p', 'rb') as fp:
                return pickle.load(fp)