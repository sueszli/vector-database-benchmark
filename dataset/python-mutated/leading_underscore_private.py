from dataclasses import dataclass
from litestar import Litestar, post
from litestar.dto import DataclassDTO

@dataclass
class Foo:
    bar: str
    _baz: str = 'Mars'

@post('/', dto=DataclassDTO[Foo], sync_to_thread=False)
def handler(data: Foo) -> Foo:
    if False:
        for i in range(10):
            print('nop')
    return data
app = Litestar(route_handlers=[handler])