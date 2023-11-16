from litestar import Litestar, get
from litestar.middleware.rate_limit import RateLimitConfig
from litestar.middleware.session.server_side import ServerSideSessionConfig
from litestar.stores.redis import RedisStore
from litestar.stores.registry import StoreRegistry
root_store = RedisStore.with_client()

@get(cache=True, sync_to_thread=False)
def cached_handler() -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'Hello, world!'
app = Litestar([cached_handler], stores=StoreRegistry(default_factory=root_store.with_namespace), middleware=[RateLimitConfig(('second', 1)).middleware, ServerSideSessionConfig().middleware])