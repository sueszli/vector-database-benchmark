from __future__ import print_function
from __future__ import division
import os
from tests.compat import unittest
from haxor_news.hacker_news import HackerNews
from haxor_news.settings import freelancer_post_id, who_is_hiring_post_id
from tests.mock_hacker_news_api import MockHackerNewsApi

class ConfigTestIntegration(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.hn = HackerNews()
        self.hn.hacker_news_api = MockHackerNewsApi()
        self.limit = len(self.hn.hacker_news_api.items)

    def test_load_hiring_and_freelance_ids(self):
        if False:
            print('Hello World!')
        self.hn.config.load_hiring_and_freelance_ids()
        assert self.hn.config.hiring_id != who_is_hiring_post_id
        assert self.hn.config.freelance_id != freelancer_post_id

    def test_load_hiring_and_freelance_ids_invalid_url(self):
        if False:
            i = 10
            return i + 15
        self.hn.config.load_hiring_and_freelance_ids(url='https://example.com')
        assert self.hn.config.hiring_id == who_is_hiring_post_id
        assert self.hn.config.freelance_id == freelancer_post_id
        os.remove('./downloaded_settings.py')

    def test_load_hiring_and_freelance_ids_from_cache_or_defaults(self):
        if False:
            i = 10
            return i + 15
        self.hn.config.load_hiring_and_freelance_ids_from_cache_or_defaults()
        assert self.hn.config.hiring_id == who_is_hiring_post_id
        assert self.hn.config.freelance_id == freelancer_post_id