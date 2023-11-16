from typing import Any, Dict
from typing_extensions import Annotated
from litestar import Litestar, get
from litestar.params import Dependency

@get('/', sync_to_thread=False)
def hello_world(optional_dependency: Annotated[int, Dependency(default=3)]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    "Notice we haven't provided the dependency to the route.\n\n    This is OK, because of the default value, and now the parameter is excluded from the docs.\n    "
    return {'hello': optional_dependency}
app = Litestar(route_handlers=[hello_world])