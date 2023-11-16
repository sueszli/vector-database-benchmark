"""Provides the redis cache service functionality."""
from __future__ import annotations
from core import feconf
from core.domain import caching_domain
import redis
from typing import Dict, List, Optional
OPPIA_REDIS_CLIENT = redis.StrictRedis(host=feconf.REDISHOST, port=feconf.REDISPORT, db=feconf.OPPIA_REDIS_DB_INDEX, decode_responses=True)
CLOUD_NDB_REDIS_CLIENT = redis.StrictRedis(host=feconf.REDISHOST, port=feconf.REDISPORT, db=feconf.CLOUD_NDB_REDIS_DB_INDEX)

def get_memory_cache_stats() -> caching_domain.MemoryCacheStats:
    if False:
        return 10
    'Returns a memory profile of the redis cache. Visit\n    https://redis.io/commands/memory-stats for more details on what exactly is\n    returned.\n\n    Returns:\n        MemoryCacheStats. MemoryCacheStats object containing the total allocated\n        memory in bytes, peak memory usage in bytes, and the total number of\n        keys stored as values.\n    '
    redis_full_profile = OPPIA_REDIS_CLIENT.memory_stats()
    memory_stats = caching_domain.MemoryCacheStats(redis_full_profile['total.allocated'], redis_full_profile['peak.allocated'], redis_full_profile['keys.count'])
    return memory_stats

def flush_caches() -> None:
    if False:
        while True:
            i = 10
    'Wipes the Redis caches clean.'
    OPPIA_REDIS_CLIENT.flushdb()
    CLOUD_NDB_REDIS_CLIENT.flushdb()

def get_multi(keys: List[str]) -> List[Optional[str]]:
    if False:
        return 10
    'Looks up a list of keys in Redis cache.\n\n    Args:\n        keys: list(str). A list of keys (strings) to look up.\n\n    Returns:\n        list(str|None). A list of values in the cache corresponding to the keys\n        that are passed in.\n    '
    assert isinstance(keys, list)
    return OPPIA_REDIS_CLIENT.mget(keys)

def set_multi(key_value_mapping: Dict[str, str]) -> bool:
    if False:
        i = 10
        return i + 15
    "Sets multiple keys' values at once in the Redis cache.\n\n    Args:\n        key_value_mapping: dict(str, str). Both the key and value are strings.\n            The value can either be a primitive binary-safe string or the\n            JSON-encoded string version of the object.\n\n    Returns:\n        bool. Whether the set action succeeded.\n    "
    assert isinstance(key_value_mapping, dict)
    return OPPIA_REDIS_CLIENT.mset(key_value_mapping)

def delete_multi(keys: List[str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Deletes multiple keys in the Redis cache.\n\n    Args:\n        keys: list(str). The keys (strings) to delete.\n\n    Returns:\n        int. Number of successfully deleted keys.\n    '
    for key in keys:
        assert isinstance(key, str)
    return OPPIA_REDIS_CLIENT.delete(*keys)