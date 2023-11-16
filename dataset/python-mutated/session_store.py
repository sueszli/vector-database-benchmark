from uuid import uuid4
import sentry_sdk
from sentry.utils.json import dumps, loads
from sentry.utils.redis import clusters
EXPIRATION_TTL = 10 * 60

class RedisSessionStore:
    """
    RedisSessionStore provides a convenience object, which when initialized will
    store attributes assigned to it into redis. The redis key is stored into
    the request session. Useful for storing data too large to be stored into
    the session cookie.

    The attributes to be backed by Redis must be declared in a subclass using
    the `redis_property` function. Do not instantiate RedisSessionStore without
    extending it to add properties. For example:

    >>> class HotDogSessionStore(RedisSessionStore):
    >>>     bun = redis_property("bun")
    >>>     condiment = redis_property("condiment")

    NOTE: Assigning attributes immediately saves their value back into the
          redis key assigned for this store. Be aware of the multiple
          round-trips implication of this.

    NOTE: This object is subject to race conditions on updating valeus as the
          entire object value is stored in one redis key.

    >>> store = RedisSessionStore(request, 'store-name')
    >>> store.regenerate()
    >>> store.some_value = 'my value'

    The value will be available across requests as long as the same same store
    name is used.

    >>> store.some_value
    'my value'

    The store may be destroyed before it expires using the ``clear`` method.

    >>> store.clear()

    It's important to note that the session store will expire if values are not
    modified within the provided ttl.
    """
    redis_namespace = 'session-cache'

    def __init__(self, request, prefix, ttl=EXPIRATION_TTL):
        if False:
            print('Hello World!')
        self.request = request
        self.prefix = prefix
        self.ttl = ttl

    @property
    def _client(self):
        if False:
            i = 10
            return i + 15
        return clusters.get('default').get_local_client_for_key(self.redis_key)

    @property
    def session_key(self):
        if False:
            print('Hello World!')
        return f'store:{self.prefix}'

    @property
    def redis_key(self):
        if False:
            i = 10
            return i + 15
        return self.request.session.get(self.session_key)

    def mark_session(self):
        if False:
            return 10
        pass

    def regenerate(self, initial_state=None):
        if False:
            while True:
                i = 10
        if initial_state is None:
            initial_state = {}
        redis_key = f'{self.redis_namespace}:{self.prefix}:{uuid4().hex}'
        self.request.session[self.session_key] = redis_key
        self.mark_session()
        value = dumps(initial_state)
        self._client.setex(redis_key, self.ttl, value)

    def clear(self):
        if False:
            while True:
                i = 10
        if not self.redis_key:
            return
        self._client.delete(self.redis_key)
        session = self.request.session
        del session[self.session_key]
        self.mark_session()

    def is_valid(self):
        if False:
            while True:
                i = 10
        return bool(self.redis_key and self.get_state() is not None)

    def get_state(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.redis_key:
            return None
        state_json = self._client.get(self.redis_key)
        if not state_json:
            return None
        try:
            return loads(state_json)
        except Exception as e:
            sentry_sdk.capture_exception(e)
        return None

def redis_property(key: str):
    if False:
        for i in range(10):
            print('nop')
    'Declare a property backed by Redis on a RedisSessionStore class.'

    def getter(store: 'RedisSessionStore'):
        if False:
            return 10
        state = store.get_state()
        try:
            return state[key] if state else None
        except KeyError as e:
            raise AttributeError(e)

    def setter(store: 'RedisSessionStore', value):
        if False:
            for i in range(10):
                print('nop')
        state = store.get_state()
        if state is None:
            return
        state[key] = value
        store._client.setex(store.redis_key, store.ttl, dumps(state))
    return property(getter, setter)