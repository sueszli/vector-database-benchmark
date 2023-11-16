from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from queue import Empty
from .websocket import PerspectiveWebsocketClient, PerspectiveWebsocketConnection, Periodic

class _StarletteTestPeriodic(Periodic):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    async def start(self):
        ...

    async def stop(self):
        ...

class _PerspectiveStarletteWebsocketConnection(PerspectiveWebsocketConnection):

    def __init__(self, client: TestClient):
        if False:
            return 10
        'A Starlette Websocket client.\n\n        NOTE: For use in tests only!\n\n        Args:\n            client (TestClient): starlette TestClient instance\n        '
        self._client = client
        self._ws = None
        self._on_message = None
        self._send_cache = None

    async def connect(self, url, on_message, max_message_size) -> None:
        self._ws = self._client.websocket_connect(url).__enter__()
        self._on_message = on_message

    def periodic(self, callback, interval) -> Periodic:
        if False:
            return 10
        return _StarletteTestPeriodic(callback=callback, interval=interval)

    def _on_message_internal(self):
        if False:
            for i in range(10):
                print('nop')
        attempt = 0
        try_count = 5
        while attempt < try_count:
            try:
                while True:
                    message = self._ws._send_queue.get(timeout=0.01)
                    if isinstance(message, BaseException):
                        raise message
                    self._ws._raise_on_close(message)
                    if 'text' in message:
                        self._on_message(message['text'])
                    if 'bytes' in message:
                        self._on_message(message['bytes'])
            except Empty:
                attempt += 1

    async def write(self, message, binary=False):
        self._on_message_internal()
        if binary:
            self._ws.send_bytes(message)
        else:
            self._ws.send_text(message)
        self._on_message_internal()

    async def close(self):
        try:
            self._ws.__exit__()
        except WebSocketDisconnect:
            return

class _PerspectiveStarletteTestClient(PerspectiveWebsocketClient):

    def __init__(self, test_client: TestClient):
        if False:
            i = 10
            return i + 15
        'Create a `PerspectiveStarletteTestClient` that interfaces with a Perspective server over a Websocket'
        super(_PerspectiveStarletteTestClient, self).__init__(_PerspectiveStarletteWebsocketConnection(test_client))

async def websocket(test_client: TestClient, url: str):
    """Create a new websocket client at the given `url` using the thread current
    tornado loop."""
    client = _PerspectiveStarletteTestClient(test_client)
    await client.connect(url)
    return client