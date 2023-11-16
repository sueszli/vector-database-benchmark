import pickle
from functools import lru_cache
from typing import Any, Dict
from django_bmemcached.memcached import BMemcached

@lru_cache(None)
def _get_bmemcached(location: str, params: bytes) -> BMemcached:
    if False:
        for i in range(10):
            print('nop')
    return BMemcached(location, pickle.loads(params))

def SingletonBMemcached(location: str, params: Dict[str, Any]) -> BMemcached:
    if False:
        while True:
            i = 10
    return _get_bmemcached(location, pickle.dumps(params))