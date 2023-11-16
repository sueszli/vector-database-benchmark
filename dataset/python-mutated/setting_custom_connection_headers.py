from litestar import Litestar, WebSocket, websocket_listener

async def accept_connection(socket: WebSocket) -> None:
    await socket.accept(headers={'Cookie': 'custom-cookie'})

@websocket_listener('/', connection_accept_handler=accept_connection)
def handler(data: str) -> str:
    if False:
        i = 10
        return i + 15
    return data
app = Litestar([handler])