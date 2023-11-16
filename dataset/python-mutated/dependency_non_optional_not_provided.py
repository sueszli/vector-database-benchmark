from typing import Any
from typing_extensions import Annotated
from litestar import Litestar, get
from litestar.params import Dependency

@get('/')
def hello_world(non_optional_dependency: Annotated[int, Dependency()]) -> dict[str, Any]:
    if False:
        print('Hello World!')
    "Notice we haven't provided the dependency to the route.\n\n    This is not great, however by explicitly marking dependencies, Litestar won't let the app start.\n    "
    return {'hello': non_optional_dependency}
app = Litestar(route_handlers=[hello_world])