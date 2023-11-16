from typing import Any, Dict
from litestar import Litestar, get
from litestar.datastructures import ImmutableState

@get('/', sync_to_thread=False)
def handler(state: ImmutableState) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    setattr(state, 'count', 1)
    return state.dict()
app = Litestar(route_handlers=[handler])