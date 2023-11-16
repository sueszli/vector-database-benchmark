import os
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
from Orange.misc.utils.embedder_utils import get_proxies, EmbedderCache

class TestProxies(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.previous_http = os.environ.get('http_proxy')
        self.previous_https = os.environ.get('https_proxy')
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        if self.previous_http is not None:
            os.environ['http_proxy'] = self.previous_http
        if self.previous_https is not None:
            os.environ['https_proxy'] = self.previous_https

    def test_add_scheme(self):
        if False:
            print('Hello World!')
        os.environ['http_proxy'] = 'test1.com'
        os.environ['https_proxy'] = 'test2.com'
        res = get_proxies()
        self.assertEqual('http://test1.com', res.get('http://'))
        self.assertEqual('http://test2.com', res.get('https://'))
        os.environ['http_proxy'] = 'test1.com/path'
        os.environ['https_proxy'] = 'test2.com/path'
        res = get_proxies()
        self.assertEqual('http://test1.com/path', res.get('http://'))
        self.assertEqual('http://test2.com/path', res.get('https://'))
        os.environ['http_proxy'] = 'https://test1.com:123'
        os.environ['https_proxy'] = 'https://test2.com:124'
        res = get_proxies()
        self.assertEqual('https://test1.com:123', res.get('http://'))
        self.assertEqual('https://test2.com:124', res.get('https://'))

    def test_both_urls(self):
        if False:
            return 10
        os.environ['http_proxy'] = 'http://test1.com:123'
        os.environ['https_proxy'] = 'https://test2.com:124'
        res = get_proxies()
        self.assertEqual('http://test1.com:123', res.get('http://'))
        self.assertEqual('https://test2.com:124', res.get('https://'))
        self.assertNotIn('all://', res)

    def test_http_only(self):
        if False:
            i = 10
            return i + 15
        os.environ['http_proxy'] = 'http://test1.com:123'
        res = get_proxies()
        self.assertEqual('http://test1.com:123', res.get('http://'))
        self.assertNotIn('https://', res)

    def test_https_only(self):
        if False:
            print('Hello World!')
        os.environ['https_proxy'] = 'https://test1.com:123'
        res = get_proxies()
        self.assertEqual('https://test1.com:123', res.get('https://'))
        self.assertNotIn('http://', res)

    def test_none(self):
        if False:
            print('Hello World!')
        ' When no variable is set return None '
        self.assertIsNone(get_proxies())

class TestEmbedderCache(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = tempfile.mkdtemp()
        patcher = patch('Orange.misc.utils.embedder_utils.cache_dir', return_value=self.temp_dir)
        patcher.start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.temp_dir)

    def test_save_load_cache(self):
        if False:
            for i in range(10):
                print('nop')
        cache = EmbedderCache('TestModel')
        self.assertDictEqual({}, cache._cache_dict)
        cache.add('abc', [1, 2, 3])
        cache.persist_cache()
        cache = EmbedderCache('TestModel')
        self.assertDictEqual({'abc': [1, 2, 3]}, cache._cache_dict)

    def test_save_cache_no_permission(self):
        if False:
            print('Hello World!')
        cache = EmbedderCache('TestModel')
        self.assertDictEqual({}, cache._cache_dict)
        cache.add('abc', [1, 2, 3])
        cache.persist_cache()
        curr_permission = os.stat(cache._cache_file_path).st_mode
        os.chmod(cache._cache_file_path, stat.S_IRUSR)
        cache.add('abcd', [1, 2, 3])
        cache.persist_cache()
        cache = EmbedderCache('TestModel')
        self.assertDictEqual({'abc': [1, 2, 3]}, cache._cache_dict)
        os.chmod(cache._cache_file_path, curr_permission)

    def test_load_cache_no_permission(self):
        if False:
            print('Hello World!')
        cache = EmbedderCache('TestModel')
        self.assertDictEqual({}, cache._cache_dict)
        cache.add('abc', [1, 2, 3])
        cache.persist_cache()
        if os.name == 'nt':
            with patch('Orange.misc.utils.embedder_utils.pickle.load', side_effect=PermissionError):
                cache = EmbedderCache('TestModel')
        else:
            os.chmod(cache._cache_file_path, stat.S_IWUSR)
            cache = EmbedderCache('TestModel')
        self.assertDictEqual({}, cache._cache_dict)

    def test_load_cache_eof_error(self):
        if False:
            return 10
        cache = EmbedderCache('TestModel')
        self.assertDictEqual({}, cache._cache_dict)
        cache.add('abc', [1, 2, 3])
        cache.persist_cache()
        with patch('Orange.misc.utils.embedder_utils.pickle.load', side_effect=EOFError):
            cache = EmbedderCache('TestModel')
            self.assertDictEqual({}, cache._cache_dict)
if __name__ == '__main__':
    unittest.main()