import os
from unittest.mock import patch
from tornado.web import url
from flower.app import rewrite_handler
from tests.unit import AsyncHTTPTestCase

class UrlsTests(AsyncHTTPTestCase):

    def test_workers_url(self):
        if False:
            while True:
                i = 10
        r = self.get('/workers')
        self.assertEqual(200, r.code)

    def test_root_url(self):
        if False:
            for i in range(10):
                print('nop')
        r = self.get('/')
        self.assertEqual(200, r.code)

    def test_tasks_api_url(self):
        if False:
            while True:
                i = 10
        with patch.dict(os.environ, {'FLOWER_UNAUTHENTICATED_API': 'true'}):
            r = self.get('/api/tasks')
            self.assertEqual(200, r.code)

class URLPrefixTests(AsyncHTTPTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.url_prefix = '/test_root'
        with self.mock_option('url_prefix', self.url_prefix):
            super().setUp()

    def test_tuple_handler_rewrite(self):
        if False:
            return 10
        r = self.get(self.url_prefix + '/workers')
        self.assertEqual(200, r.code)

    def test_root_url(self):
        if False:
            i = 10
            return i + 15
        r = self.get(self.url_prefix + '/')
        self.assertEqual(200, r.code)

    def test_tasks_api_url(self):
        if False:
            i = 10
            return i + 15
        with patch.dict(os.environ, {'FLOWER_UNAUTHENTICATED_API': 'true'}):
            r = self.get(self.url_prefix + '/api/tasks')
            self.assertEqual(200, r.code)

    def test_base_url_no_longer_working(self):
        if False:
            print('Hello World!')
        r = self.get('/')
        self.assertEqual(404, r.code)

class RewriteHandlerTests(AsyncHTTPTestCase):

    def target(self):
        if False:
            return 10
        return None

    def test_url_rewrite_using_URLSpec(self):
        if False:
            print('Hello World!')
        old_handler = url('/', self.target, name='test')
        new_handler = rewrite_handler(old_handler, 'test_root')
        self.assertIsInstance(new_handler, url)
        self.assertTrue(new_handler.regex.match('/test_root/'))
        self.assertFalse(new_handler.regex.match('/'))
        self.assertFalse(new_handler.regex.match('/'))

    def test_url_rewrite_using_tuple(self):
        if False:
            while True:
                i = 10
        old_handler = ('/', self.target)
        new_handler = rewrite_handler(old_handler, 'test_root')
        self.assertIsInstance(new_handler, tuple)
        self.assertEqual(new_handler[0], '/test_root/')