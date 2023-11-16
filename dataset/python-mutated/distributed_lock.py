from mage_ai.services.redis.redis import init_redis_client
from mage_ai.settings import REDIS_URL

class DistributedLock:

    def __init__(self, lock_key_prefix='LOCK_KEY', lock_timeout=30):
        if False:
            for i in range(10):
                print('nop')
        self.lock_key_prefix = lock_key_prefix
        self.lock_timeout = lock_timeout
        self.redis_client = init_redis_client(REDIS_URL)

    def __lock_key(self, key) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.lock_key_prefix}_{key}'

    def try_acquire_lock(self, key, timeout: int=None) -> bool:
        if False:
            print('Hello World!')
        if not self.redis_client:
            return True
        acquired = self.redis_client.set(self.__lock_key(key), '1', nx=True, ex=timeout or self.lock_timeout)
        return acquired is True

    def release_lock(self, key):
        if False:
            for i in range(10):
                print('nop')
        if self.redis_client:
            self.redis_client.delete(self.__lock_key(key))