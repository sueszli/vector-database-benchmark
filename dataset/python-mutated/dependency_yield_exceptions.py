from typing import Dict, Generator
from litestar import Litestar, get
from litestar.di import Provide
STATE = {'result': None, 'connection': 'closed'}

def generator_function() -> Generator[str, None, None]:
    if False:
        i = 10
        return i + 15
    'Set the connection state to open and close it after the handler returns.\n\n    If an error occurs, set `result` to `"error"`, else set it to `"OK"`.\n    '
    try:
        STATE['connection'] = 'open'
        yield 'hello'
        STATE['result'] = 'OK'
    except ValueError:
        STATE['result'] = 'error'
    finally:
        STATE['connection'] = 'closed'

@get('/{name:str}', dependencies={'message': Provide(generator_function)})
def index(name: str, message: str) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    'If `name` is "John", return a message, otherwise raise an error.'
    if name == 'John':
        return {name: message}
    raise ValueError()
app = Litestar(route_handlers=[index])