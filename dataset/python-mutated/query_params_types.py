from datetime import datetime, timedelta
from typing import Any, Dict, List
from litestar import Litestar, get

@get('/', sync_to_thread=False)
def index(date: datetime, number: int, floating_number: float, strings: List[str]) -> Dict[str, Any]:
    if False:
        return 10
    return {'datetime': date + timedelta(days=1), 'int': number, 'float': floating_number, 'list': strings}
app = Litestar(route_handlers=[index])