import logging
import pytest
import salt.cache
from tests.pytests.functional.cache.helpers import run_common_cache_tests
log = logging.getLogger(__name__)

@pytest.fixture
def cache(minion_opts):
    if False:
        while True:
            i = 10
    opts = minion_opts.copy()
    opts['memcache_expire_seconds'] = 42
    cache = salt.cache.factory(opts)
    return cache

def test_caching(subtests, cache):
    if False:
        i = 10
        return i + 15
    run_common_cache_tests(subtests, cache)