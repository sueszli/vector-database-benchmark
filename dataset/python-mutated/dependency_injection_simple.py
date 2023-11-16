from litestar import Litestar, websocket_listener
from litestar.di import Provide

def some_dependency() -> str:
    if False:
        print('Hello World!')
    return 'hello'

@websocket_listener('/', dependencies={'some': Provide(some_dependency)})
async def handler(data: str, some: str) -> str:
    return data + some
app = Litestar([handler])