"""Block blocking calls being done in asyncio."""
from http.client import HTTPConnection
import time
from .util.async_ import protect_loop

def enable() -> None:
    if False:
        print('Hello World!')
    'Enable the detection of blocking calls in the event loop.'
    HTTPConnection.putrequest = protect_loop(HTTPConnection.putrequest)
    time.sleep = protect_loop(time.sleep, strict=False)