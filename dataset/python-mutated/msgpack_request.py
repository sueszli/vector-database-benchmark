from typing import Any, Dict
from typing_extensions import Annotated
from litestar import Litestar, post
from litestar.enums import RequestEncodingType
from litestar.params import Body

@post(path='/', sync_to_thread=False)
def msgpack_handler(data: Annotated[Dict[str, Any], Body(media_type=RequestEncodingType.MESSAGEPACK)]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return data
app = Litestar(route_handlers=[msgpack_handler])