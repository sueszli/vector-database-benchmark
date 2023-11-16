from dataclasses import dataclass
from litestar import Litestar, post
from litestar.dto import DataclassDTO, DTOConfig

@dataclass
class Foo:
    bar: str
    _baz: str = 'Mars'

class DTO(DataclassDTO[Foo]):
    config = DTOConfig(underscore_fields_private=False)

@post('/', dto=DTO, sync_to_thread=False)
def handler(data: Foo) -> Foo:
    if False:
        i = 10
        return i + 15
    return data
app = Litestar(route_handlers=[handler])