from litestar import Litestar, WebSocket
from litestar.handlers import WebsocketListener

class Handler(WebsocketListener):
    path = '/'

    def on_accept(self, socket: WebSocket) -> None:
        if False:
            for i in range(10):
                print('nop')
        print('Connection accepted')

    def on_disconnect(self, socket: WebSocket) -> None:
        if False:
            print('Hello World!')
        print('Connection closed')

    def on_receive(self, data: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return data
app = Litestar([Handler])