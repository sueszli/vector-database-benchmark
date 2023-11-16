from typing import Optional
from .. import context
from ..awaitable_response import AwaitableResponse
from ..logging import log

def run_javascript(code: str, *, respond: Optional[bool]=None, timeout: float=1.0, check_interval: float=0.01) -> AwaitableResponse:
    if False:
        return 10
    'Run JavaScript\n\n    This function runs arbitrary JavaScript code on a page that is executed in the browser.\n    The client must be connected before this function is called.\n    To access a client-side object by ID, use the JavaScript function `getElement()`.\n\n    If the function is awaited, the result of the JavaScript code is returned.\n    Otherwise, the JavaScript code is executed without waiting for a response.\n\n    :param code: JavaScript code to run\n    :param timeout: timeout in seconds (default: `1.0`)\n    :param check_interval: interval in seconds to check for a response (default: `0.01`)\n\n    :return: AwaitableResponse that can be awaited to get the result of the JavaScript code\n    '
    if respond is True:
        log.warning('The "respond" argument of run_javascript() has been removed. Now the function always returns an AwaitableResponse that can be awaited. Please remove the "respond=True" argument.')
    if respond is False:
        raise ValueError('The "respond" argument of run_javascript() has been removed. Now the function always returns an AwaitableResponse that can be awaited. Please remove the "respond=False" argument and call the function without awaiting.')
    client = context.get_client()
    if not client.has_socket_connection:
        raise RuntimeError('Cannot run JavaScript before client is connected; try "await client.connected()" or "client.on_connect(...)".')
    return client.run_javascript(code, timeout=timeout, check_interval=check_interval)