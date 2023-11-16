from typing import Any, AsyncGenerator, Dict, Generator
from elastic_transport import Transport
from elasticsearch8 import AsyncElasticsearch, Elasticsearch
from elasticsearch8.helpers import async_bulk, async_reindex, async_scan, async_streaming_bulk, bulk, reindex, scan, streaming_bulk
es = Elasticsearch([{'host': 'localhost', 'port': 9443}], transport_class=Transport, sniff_on_start=True, sniffer_timeout=0.1, sniff_timeout=1, sniff_on_connection_fail=False, max_retries=1, retry_on_status={100, 400, 503}, retry_on_timeout=True)

def sync_gen() -> Generator[Dict[Any, Any], None, None]:
    if False:
        while True:
            i = 10
    yield {}

def scan_types() -> None:
    if False:
        return 10
    for _ in scan(es, query={'query': {'match_all': {}}}, request_timeout=10, clear_scroll=True, scroll_kwargs={'request_timeout': 10}):
        pass
    for _ in scan(es, raise_on_error=False, preserve_order=False, scroll='10m', size=10, request_timeout=10.0):
        pass

def streaming_bulk_types() -> None:
    if False:
        print('Hello World!')
    for _ in streaming_bulk(es, sync_gen()):
        pass
    for _ in streaming_bulk(es, sync_gen().__iter__()):
        pass
    for _ in streaming_bulk(es, [{'key': 'value'}]):
        pass

def bulk_types() -> None:
    if False:
        return 10
    (_, _) = bulk(es, sync_gen())
    (_, _) = bulk(es, sync_gen().__iter__())
    (_, _) = bulk(es, [{'key': 'value'}])

def reindex_types() -> None:
    if False:
        while True:
            i = 10
    (_, _) = reindex(es, 'src-index', 'target-index', query={'query': {'match': {'key': 'val'}}})
    (_, _) = reindex(es, source_index='src-index', target_index='target-index', target_client=es)
    (_, _) = reindex(es, 'src-index', 'target-index', chunk_size=1, scroll='10m', scan_kwargs={'request_timeout': 10}, bulk_kwargs={'request_timeout': 10})
es2 = AsyncElasticsearch([{'host': 'localhost', 'port': 9443}], sniff_on_start=True, sniffer_timeout=0.1, sniff_timeout=1, sniff_on_connection_fail=False, max_retries=1, retry_on_status={100, 400, 503}, retry_on_timeout=True)

async def async_gen() -> AsyncGenerator[Dict[Any, Any], None]:
    yield {}

async def async_scan_types() -> None:
    async for _ in async_scan(es2, query={'query': {'match_all': {}}}, request_timeout=10, clear_scroll=True, scroll_kwargs={'request_timeout': 10}):
        pass
    async for _ in async_scan(es2, raise_on_error=False, preserve_order=False, scroll='10m', size=10, request_timeout=10.0):
        pass

async def async_streaming_bulk_types() -> None:
    async for _ in async_streaming_bulk(es2, async_gen()):
        pass
    async for _ in async_streaming_bulk(es2, async_gen().__aiter__()):
        pass
    async for _ in async_streaming_bulk(es2, [{'key': 'value'}]):
        pass

async def async_bulk_types() -> None:
    (_, _) = await async_bulk(es2, async_gen())
    (_, _) = await async_bulk(es2, async_gen().__aiter__())
    (_, _) = await async_bulk(es2, [{}])

async def async_reindex_types() -> None:
    (_, _) = await async_reindex(es2, 'src-index', 'target-index', query={'query': {'match': {'key': 'val'}}})
    (_, _) = await async_reindex(es2, source_index='src-index', target_index='target-index', target_client=es2)
    (_, _) = await async_reindex(es2, 'src-index', 'target-index', chunk_size=1, scroll='10m', scan_kwargs={'request_timeout': 10}, bulk_kwargs={'request_timeout': 10})