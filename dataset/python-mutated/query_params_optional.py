from typing import Dict, Optional
from litestar import Litestar, get

@get('/', sync_to_thread=False)
def index(param: Optional[str]=None) -> Dict[str, Optional[str]]:
    if False:
        while True:
            i = 10
    return {'param': param}
app = Litestar(route_handlers=[index])