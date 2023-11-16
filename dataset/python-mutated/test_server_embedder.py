import asyncio
import unittest
from random import random
from unittest.mock import MagicMock, call, patch
import numpy as np
from httpx import ReadTimeout
from Orange.data import Domain, StringVariable, Table
from Orange.misc.tests.example_embedder import ExampleServerEmbedder
_HTTPX_POST_METHOD = 'httpx.AsyncClient.post'

class AsyncMock(MagicMock):

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

class DummyResponse:

    def __init__(self, content):
        if False:
            return 10
        self.content = content

def make_dummy_post(response):
    if False:
        return 10

    @staticmethod
    async def dummy_post(url, headers, content=None, data=None):
        assert (content is None) ^ (data is None)
        await asyncio.sleep(random() / 10)
        return DummyResponse(content=response)
    return dummy_post
regular_dummy_sr = make_dummy_post(b'{"embedding": [0, 1]}')

class TestServerEmbedder(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.embedder = ExampleServerEmbedder('test', 10, 'https://test.com', 'image')
        self.embedder.clear_cache()
        self.test_data = Table.from_numpy(Domain([], metas=[StringVariable('test_var')]), np.empty((3, 0)), metas=np.array([['test1'], ['test2'], ['test3']]))

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_responses(self):
        if False:
            while True:
                i = 10
        results = self.embedder.embedd_data(self.test_data)
        np.testing.assert_array_equal(results, [[0, 1]] * 3)
        self.assertEqual(3, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b''))
    def test_responses_empty(self):
        if False:
            print('Hello World!')
        results = self.embedder.embedd_data(self.test_data)
        self.assertListEqual([None] * 3, results)
        self.assertEqual(0, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b'blabla'))
    def test_on_non_json_response(self):
        if False:
            i = 10
            return i + 15
        results = self.embedder.embedd_data(self.test_data)
        self.assertListEqual([None] * 3, results)
        self.assertEqual(0, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b'{"wrong-key": [0, 1]}'))
    def test_on_json_wrong_key_response(self):
        if False:
            for i in range(10):
                print('nop')
        results = self.embedder.embedd_data(self.test_data)
        self.assertListEqual([None] * 3, results)
        self.assertEqual(0, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_persistent_caching(self):
        if False:
            return 10
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)
        self.embedder = ExampleServerEmbedder('test', 10, 'https://test.com', 'image')
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)
        self.embedder.clear_cache()
        self.embedder = ExampleServerEmbedder('test', 10, 'https://test.com', 'image')
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_different_models_caches(self):
        if False:
            print('Hello World!')
        self.embedder.clear_cache()
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)
        self.embedder = ExampleServerEmbedder('different_emb', 10, 'https://test.com', 'image')
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)
        self.embedder = ExampleServerEmbedder('test', 10, 'https://test.com', 'image')
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)
        self.embedder.clear_cache()

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_too_many_examples_for_one_batch(self):
        if False:
            return 10
        test_data = Table.from_numpy(Domain([], metas=[StringVariable('test_var')]), np.empty((200, 0)), metas=np.array([[f'test{i}'] for i in range(200)]))
        results = self.embedder.embedd_data(test_data)
        np.testing.assert_array_equal(results, [[0, 1]] * 200)
        self.assertEqual(200, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, side_effect=OSError)
    def test_connection_error(self, _):
        if False:
            return 10
        for num_rows in range(1, 20):
            test_data = Table.from_numpy(Domain([], metas=[StringVariable('test_var')]), np.empty((num_rows, 0)), metas=np.array([[f'test{i}'] for i in range(num_rows)]))
            with self.assertRaises(ConnectionError):
                self.embedder.embedd_data(test_data)
            self.setUp()

    @patch(_HTTPX_POST_METHOD, side_effect=ReadTimeout('', request=None))
    def test_read_error(self, _):
        if False:
            print('Hello World!')
        for num_rows in range(1, 20):
            test_data = Table.from_numpy(Domain([], metas=[StringVariable('test_var')]), np.empty((num_rows, 0)), metas=np.array([[f'test{i}'] for i in range(num_rows)]))
            with self.assertRaises(ConnectionError):
                self.embedder.embedd_data(test_data)
            self.setUp()

    @patch(_HTTPX_POST_METHOD, side_effect=ValueError)
    def test_other_errors(self, _):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self.embedder.embedd_data(self.test_data)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_encode_data_instance(self):
        if False:
            for i in range(10):
                print('nop')
        mocked_fun = self.embedder._encode_data_instance = AsyncMock(return_value=b'abc')
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(3, mocked_fun.call_count)
        mocked_fun.assert_has_calls([call(item) for item in self.test_data], any_order=True)

    @patch(_HTTPX_POST_METHOD, return_value=DummyResponse(b''), new_callable=AsyncMock)
    def test_retries(self, mock):
        if False:
            for i in range(10):
                print('nop')
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(len(self.test_data) * 3, mock.call_count)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_callback(self):
        if False:
            return 10
        mock = MagicMock()
        self.embedder.embedd_data(self.test_data, callback=mock)
        process_items = [call(x) for x in np.linspace(0, 1, len(self.test_data))]
        mock.assert_has_calls(process_items)
if __name__ == '__main__':
    unittest.main()