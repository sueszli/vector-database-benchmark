import datetime
import logging
from luigi.target import Target
from luigi.parameter import Parameter
logger = logging.getLogger('luigi-interface')
try:
    import redis
except ImportError:
    logger.warning('Loading redis_store module without redis installed. Will crash at runtime if redis_store functionality is used.')

class RedisTarget(Target):
    """ Target for a resource in Redis."""
    marker_prefix = Parameter(default='luigi', config_path=dict(section='redis', name='marker-prefix'))

    def __init__(self, host, port, db, update_id, password=None, socket_timeout=None, expire=None):
        if False:
            i = 10
            return i + 15
        '\n        :param host: Redis server host\n        :type host: str\n        :param port: Redis server port\n        :type port: int\n        :param db: database index\n        :type db: int\n        :param update_id: an identifier for this data hash\n        :type update_id: str\n        :param password: a password to connect to the redis server\n        :type password: str\n        :param socket_timeout: client socket timeout\n        :type socket_timeout: int\n        :param expire: timeout before the target is deleted\n        :type expire: int\n\n        '
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout
        self.update_id = update_id
        self.expire = expire
        self.redis_client = redis.StrictRedis(host=self.host, port=self.port, password=self.password, db=self.db, socket_timeout=self.socket_timeout)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.marker_key()

    def marker_key(self):
        if False:
            while True:
                i = 10
        '\n        Generate a key for the indicator hash.\n        '
        return '%s:%s' % (self.marker_prefix, self.update_id)

    def touch(self):
        if False:
            return 10
        '\n        Mark this update as complete.\n\n        We index the parameters `update_id` and `date`.\n        '
        marker_key = self.marker_key()
        self.redis_client.hset(marker_key, 'update_id', self.update_id)
        self.redis_client.hset(marker_key, 'date', datetime.datetime.now().isoformat())
        if self.expire is not None:
            self.redis_client.expire(marker_key, self.expire)

    def exists(self):
        if False:
            while True:
                i = 10
        '\n        Test, if this task has been run.\n        '
        return self.redis_client.exists(self.marker_key()) == 1