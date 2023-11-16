from typing import Dict
from litestar import Litestar, get

@get('/', sync_to_thread=False)
def index(param: str='hello') -> Dict[str, str]:
    if False:
        print('Hello World!')
    return {'param': param}
app = Litestar(route_handlers=[index])