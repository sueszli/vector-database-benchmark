from uvicorn._types import ASGI2Application, ASGIReceiveCallable, ASGISendCallable, Scope

class ASGI2Middleware:

    def __init__(self, app: 'ASGI2Application'):
        if False:
            print('Hello World!')
        self.app = app

    async def __call__(self, scope: 'Scope', receive: 'ASGIReceiveCallable', send: 'ASGISendCallable') -> None:
        instance = self.app(scope)
        await instance(receive, send)