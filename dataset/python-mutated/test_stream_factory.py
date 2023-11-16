import functools
import json
import time
from pathlib import Path
import pytest
import requests
from source_zoho_crm.streams import IncrementalZohoCrmStream, ZohoStreamFactory
HERE = Path(__file__).parent

@pytest.fixture()
def config():
    if False:
        return 10
    with open(HERE.parent / 'secrets/config.json', 'r') as file:
        return json.loads(file.read())

@pytest.fixture
def request_sniffer(mocker):
    if False:
        return 10

    def request_decorator(stats):
        if False:
            i = 10
            return i + 15

        def decorator(func):
            if False:
                return 10

            def inner(*args, **kwargs):
                if False:
                    return 10
                resp = func(*args, **kwargs)
                stats[resp.url.split('/crm', 1)[-1]] = resp.status_code
                return resp
            return inner
        return decorator
    stats = {}
    decorated = request_decorator(stats)
    mocker.patch('source_zoho_crm.api.requests.request', decorated(requests.request))
    mocker.patch('source_zoho_crm.api.requests.get', decorated(requests.get))
    return stats

def timeit(func):
    if False:
        print('Hello World!')

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total_time = int(end - start)
        print(f'{func}.__name__ execution took {total_time} seconds')
        assert total_time <= 15
        return result
    return wrapper

def test_stream_factory(request_sniffer, config):
    if False:
        i = 10
        return i + 15
    factory = ZohoStreamFactory(config)
    streams = timeit(factory.produce)()
    expected_stream_names = {'Accounts', 'Activities', 'Attachments', 'Calls', 'Campaigns', 'Cases', 'Contacts', 'Deals', 'Events', 'Invoices', 'Leads', 'Notes', 'Price_Books', 'Products', 'Purchase_Orders', 'Quotes', 'Sales_Orders', 'Solutions', 'Tasks', 'Vendors'}
    stream_names = set()
    for stream in streams:
        assert stream.supports_incremental
        assert isinstance(stream, IncrementalZohoCrmStream)
        assert stream.primary_key
        assert stream.url_base
        assert stream.path()
        assert stream.get_json_schema()
        stream_names.add(stream.module.api_name)
    assert expected_stream_names.issubset(stream_names)
    (expected_stream_names, unexpected_stream_names) = (set(), set())
    for (url, status) in request_sniffer.items():
        assert status in (200, 204)
        module = url.split('?module=')[-1]
        if module == url:
            module = url.split('modules/')[-1]
            if module == url:
                continue
        expected_stream_names.add(module)
        if status == 204:
            unexpected_stream_names.add(module)
    assert expected_stream_names - unexpected_stream_names == stream_names