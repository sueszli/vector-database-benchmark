from typing import Any, Dict
from litestar import Litestar, get
from litestar.datastructures import State

class MyState(State):
    count: int = 0

    def increment(self) -> None:
        if False:
            i = 10
            return i + 15
        self.count += 1

@get('/', sync_to_thread=False)
def handler(state: MyState) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    state.increment()
    return state.dict()
app = Litestar(route_handlers=[handler])