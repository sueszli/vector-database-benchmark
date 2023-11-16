from typing import Any, Dict, Generator
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, reindex, scan, streaming_bulk
es = Elasticsearch([{'host': 'localhost', 'port': 9443}], sniff_on_start=True, sniffer_timeout=0.1, sniff_timeout=1, sniff_on_connection_fail=False, max_retries=1, retry_on_status={100, 400, 503}, retry_on_timeout=True)
es.options(request_timeout=1.0, max_retries=0, api_key='api-key-example').search(index='test-index')
es.options(request_timeout=1.0, max_retries=0, api_key='api-key-example').indices.exists(index='test-index')

def sync_gen() -> Generator[Dict[Any, Any], None, None]:
    if False:
        print('Hello World!')
    yield {}

def scan_types() -> None:
    if False:
        while True:
            i = 10
    for _ in scan(es, query={'query': {'match_all': {}}}, request_timeout=10, clear_scroll=True, scroll_kwargs={'request_timeout': 10}):
        pass
    for _ in scan(es, raise_on_error=False, preserve_order=False, scroll='10m', size=10, request_timeout=10.0):
        pass

def streaming_bulk_types() -> None:
    if False:
        return 10
    for _ in streaming_bulk(es, sync_gen()):
        pass
    for _ in streaming_bulk(es, sync_gen().__iter__()):
        pass
    for _ in streaming_bulk(es, [{}]):
        pass

def bulk_types() -> None:
    if False:
        print('Hello World!')
    (_, _) = bulk(es, sync_gen())
    (_, _) = bulk(es, sync_gen().__iter__())
    (_, _) = bulk(es, [{}])
    (_, _) = bulk(es, ({'key': 'value'},))

def reindex_types() -> None:
    if False:
        i = 10
        return i + 15
    (_, _) = reindex(es, 'src-index', 'target-index', query={'query': {'match': {'key': 'val'}}})
    (_, _) = reindex(es, source_index='src-index', target_index='target-index', target_client=es)
    (_, _) = reindex(es, 'src-index', 'target-index', chunk_size=1, scroll='10m', scan_kwargs={'request_timeout': 10}, bulk_kwargs={'request_timeout': 10})