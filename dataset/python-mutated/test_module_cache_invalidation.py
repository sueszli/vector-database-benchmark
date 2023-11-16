import logging
import synapse
from synapse.module_api import cached
from tests.replication._base import BaseMultiWorkerStreamTestCase
logger = logging.getLogger(__name__)
FIRST_VALUE = 'one'
SECOND_VALUE = 'two'
KEY = 'mykey'

class TestCache:
    current_value = FIRST_VALUE

    @cached()
    async def cached_function(self, user_id: str) -> str:
        return self.current_value

class ModuleCacheInvalidationTestCase(BaseMultiWorkerStreamTestCase):
    servlets = [synapse.rest.admin.register_servlets]

    def test_module_cache_full_invalidation(self) -> None:
        if False:
            print('Hello World!')
        main_cache = TestCache()
        self.hs.get_module_api().register_cached_function(main_cache.cached_function)
        worker_hs = self.make_worker_hs('synapse.app.generic_worker')
        worker_cache = TestCache()
        worker_hs.get_module_api().register_cached_function(worker_cache.cached_function)
        self.assertEqual(FIRST_VALUE, self.get_success(main_cache.cached_function(KEY)))
        self.assertEqual(FIRST_VALUE, self.get_success(worker_cache.cached_function(KEY)))
        main_cache.current_value = SECOND_VALUE
        worker_cache.current_value = SECOND_VALUE
        self.assertEqual(FIRST_VALUE, self.get_success(main_cache.cached_function(KEY)))
        self.assertEqual(FIRST_VALUE, self.get_success(worker_cache.cached_function(KEY)))
        self.get_success(self.hs.get_module_api().invalidate_cache(main_cache.cached_function, (KEY,)))
        self.assertEqual(SECOND_VALUE, self.get_success(main_cache.cached_function(KEY)))
        self.assertEqual(SECOND_VALUE, self.get_success(worker_cache.cached_function(KEY)))