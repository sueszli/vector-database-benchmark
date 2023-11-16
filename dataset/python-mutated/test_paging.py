from azure.core.paging import ItemPaged
from azure.core.exceptions import HttpResponseError
import pytest

class TestPaging(object):

    def test_basic_paging(self):
        if False:
            return 10

        def get_next(continuation_token=None):
            if False:
                for i in range(10):
                    print('nop')
            'Simplify my life and return JSON and not response, but should be response.'
            if not continuation_token:
                return {'nextLink': 'page2', 'value': ['value1.0', 'value1.1']}
            else:
                return {'nextLink': None, 'value': ['value2.0', 'value2.1']}

        def extract_data(response):
            if False:
                print('Hello World!')
            return (response['nextLink'], iter(response['value']))
        pager = ItemPaged(get_next, extract_data)
        result_iterated = list(pager)
        assert ['value1.0', 'value1.1', 'value2.0', 'value2.1'] == result_iterated

    def test_by_page_paging(self):
        if False:
            while True:
                i = 10

        def get_next(continuation_token=None):
            if False:
                print('Hello World!')
            'Simplify my life and return JSON and not response, but should be response.'
            if not continuation_token:
                return {'nextLink': 'page2', 'value': ['value1.0', 'value1.1']}
            else:
                return {'nextLink': None, 'value': ['value2.0', 'value2.1']}

        def extract_data(response):
            if False:
                return 10
            return (response['nextLink'], iter(response['value']))
        pager = ItemPaged(get_next, extract_data).by_page()
        page1 = next(pager)
        assert list(page1) == ['value1.0', 'value1.1']
        page2 = next(pager)
        assert list(page2) == ['value2.0', 'value2.1']
        with pytest.raises(StopIteration):
            next(pager)

    def test_advance_paging(self):
        if False:
            i = 10
            return i + 15

        def get_next(continuation_token=None):
            if False:
                print('Hello World!')
            'Simplify my life and return JSON and not response, but should be response.'
            if not continuation_token:
                return {'nextLink': 'page2', 'value': ['value1.0', 'value1.1']}
            else:
                return {'nextLink': None, 'value': ['value2.0', 'value2.1']}

        def extract_data(response):
            if False:
                while True:
                    i = 10
            return (response['nextLink'], iter(response['value']))
        pager = ItemPaged(get_next, extract_data)
        page1 = next(pager)
        assert page1 == 'value1.0'
        page1 = next(pager)
        assert page1 == 'value1.1'
        page2 = next(pager)
        assert page2 == 'value2.0'
        page2 = next(pager)
        assert page2 == 'value2.1'
        with pytest.raises(StopIteration):
            next(pager)

    def test_none_value(self):
        if False:
            while True:
                i = 10

        def get_next(continuation_token=None):
            if False:
                while True:
                    i = 10
            return {'nextLink': None, 'value': None}

        def extract_data(response):
            if False:
                while True:
                    i = 10
            return (response['nextLink'], iter(response['value'] or []))
        pager = ItemPaged(get_next, extract_data)
        result_iterated = list(pager)
        assert len(result_iterated) == 0

    def test_print(self):
        if False:
            while True:
                i = 10

        def get_next(continuation_token=None):
            if False:
                return 10
            return {'nextLink': None, 'value': None}

        def extract_data(response):
            if False:
                i = 10
                return i + 15
            return (response['nextLink'], iter(response['value'] or []))
        pager = ItemPaged(get_next, extract_data)
        output = repr(pager)
        assert output.startswith('<iterator object azure.core.paging.ItemPaged at')

    def test_paging_continue_on_error(self):
        if False:
            while True:
                i = 10

        def get_next(continuation_token=None):
            if False:
                print('Hello World!')
            if not continuation_token:
                return {'nextLink': 'foo', 'value': ['bar']}
            else:
                raise HttpResponseError()

        def extract_data(response):
            if False:
                i = 10
                return i + 15
            return (response['nextLink'], iter(response['value'] or []))
        pager = ItemPaged(get_next, extract_data)
        assert next(pager) == 'bar'
        with pytest.raises(HttpResponseError) as err:
            next(pager)
        assert err.value.continuation_token == 'foo'