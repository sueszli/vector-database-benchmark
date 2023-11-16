from typing import Any, Dict
from litestar import Litestar, get

@get('/', sync_to_thread=False)
def hello_world(optional_dependency: int=3) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    "Notice we haven't provided the dependency to the route.\n\n    This is OK, because of the default value, but the parameter shows in the docs.\n    "
    return {'hello': optional_dependency}
app = Litestar(route_handlers=[hello_world])