"""
Some test cases for JSON api runner
"""
from unittest import TestCase
from urllib.parse import urlencode, urljoin
from redash.query_runner.json_ds import JSON

def mock_api(url, method, **request_options):
    if False:
        for i in range(10):
            print('nop')
    if 'params' in request_options:
        qs = urlencode(request_options['params'])
        url = urljoin(url, '?{}'.format(qs))
    (data, error) = (None, None)
    if url == 'http://localhost/basics':
        data = [{'id': 1}, {'id': 2}]
    elif url == 'http://localhost/token-test':
        data = {'next_page_token': '2', 'records': [{'id': 1}, {'id': 2}]}
    elif url == 'http://localhost/token-test?page_token=2':
        data = {'next_page_token': '3', 'records': [{'id': 3}, {'id': 4}]}
    elif url == 'http://localhost/token-test?page_token=3':
        data = {'records': [{'id': 5}]}
    elif url == 'http://localhost/hateoas':
        data = {'_embedded': {'records': [{'id': 10}, {'id': 11}]}, '_links': {'first': {'href': 'http://localhost/hateoas'}, 'self': {'href': 'http://localhost/hateoas'}, 'next': {'href': 'http://localhost/hateoas?page=2'}, 'last': {'href': 'http://localhost/hateoas?page=2'}}, 'page': {'size': 2, 'totalElements': 3, 'totalPages': 2}}
    elif url == 'http://localhost/hateoas?page=2':
        data = {'_embedded': {'records': [{'id': 12}]}, '_links': {'first': {'href': 'http://localhost/hateoas'}, 'self': {'href': 'http://localhost/hateoas?page=2'}, 'prev': {'href': 'http://localhost/hateoas'}, 'last': {'href': 'http://localhost/hateoas?page=2'}}, 'page': {'size': 2, 'totalElements': 3, 'totalPages': 2}}
    else:
        error = '404: {} not found'.format(url)
    return (data, error)

class TestJSON(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.runner = JSON({'base_url': 'http://localhost/'})
        self.runner._get_json_response = mock_api

    def test_basics(self):
        if False:
            while True:
                i = 10
        q = {'url': 'basics'}
        (results, error) = self.runner._run_json_query(q)
        expected = [{'id': 1}, {'id': 2}]
        self.assertEqual(results['rows'], expected)

    def test_token_pagination(self):
        if False:
            while True:
                i = 10
        q = {'url': 'token-test', 'pagination': {'type': 'token', 'fields': ['next_page_token', 'page_token']}, 'path': 'records'}
        (results, error) = self.runner._run_json_query(q)
        self.assertIsNone(error)
        expected = [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}]
        self.assertEqual(results['rows'], expected)

    def test_url_pagination(self):
        if False:
            while True:
                i = 10
        q = {'url': 'hateoas', 'pagination': {'type': 'url', 'path': '_links.next.href'}, 'path': '_embedded.records', 'fields': ['id']}
        (results, error) = self.runner._run_json_query(q)
        self.assertIsNone(error)
        expected = [{'id': 10}, {'id': 11}, {'id': 12}]
        self.assertEqual(results['rows'], expected)