from time import sleep
from helpers import unittest
import pytest
try:
    import redis
except ImportError:
    raise unittest.SkipTest('Unable to load redis module')
from luigi.contrib.redis_store import RedisTarget
HOST = 'localhost'
PORT = 6379
DB = 15
PASSWORD = None
SOCKET_TIMEOUT = None
MARKER_PREFIX = 'luigi_test'
EXPIRE = 5

@pytest.mark.contrib
class RedisTargetTest(unittest.TestCase):
    """ Test touch, exists and target expiration"""

    def test_touch_and_exists(self):
        if False:
            i = 10
            return i + 15
        target = RedisTarget(HOST, PORT, DB, 'update_id', PASSWORD)
        target.marker_prefix = MARKER_PREFIX
        flush()
        self.assertFalse(target.exists(), 'Target should not exist before touching it')
        target.touch()
        self.assertTrue(target.exists(), 'Target should exist after touching it')
        flush()

    def test_expiration(self):
        if False:
            for i in range(10):
                print('nop')
        target = RedisTarget(HOST, PORT, DB, 'update_id', PASSWORD, None, EXPIRE)
        target.marker_prefix = MARKER_PREFIX
        flush()
        target.touch()
        self.assertTrue(target.exists(), 'Target should exist after touching it and before expiring')
        sleep(EXPIRE)
        self.assertFalse(target.exists(), 'Target should not exist after expiring')
        flush()

def flush():
    if False:
        print('Hello World!')
    ' Flush test DB'
    redis_client = redis.StrictRedis(host=HOST, port=PORT, db=DB, socket_timeout=SOCKET_TIMEOUT)
    redis_client.flushdb()