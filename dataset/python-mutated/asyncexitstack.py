from typing import Optional
from prefect._vendor.fastapi.concurrency import AsyncExitStack
from starlette.types import ASGIApp, Receive, Scope, Send

class AsyncExitStackMiddleware:

    def __init__(self, app: ASGIApp, context_name: str='fastapi_astack') -> None:
        if False:
            i = 10
            return i + 15
        self.app = app
        self.context_name = context_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        dependency_exception: Optional[Exception] = None
        async with AsyncExitStack() as stack:
            scope[self.context_name] = stack
            try:
                await self.app(scope, receive, send)
            except Exception as e:
                dependency_exception = e
                raise e
        if dependency_exception:
            raise dependency_exception