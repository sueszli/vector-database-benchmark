import httpx
import pytest

from tests.middleware.test_logging import caplog_for_logger
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.middleware.message_logger import MessageLoggerMiddleware


@pytest.mark.anyio
async def test_message_logger(caplog):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    with caplog_for_logger(caplog, "uvicorn.asgi"):
        caplog.set_level(TRACE_LOG_LEVEL, logger="uvicorn.asgi")
        caplog.set_level(TRACE_LOG_LEVEL)

        app = MessageLoggerMiddleware(app)
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/")
        assert response.status_code == 200
        messages = [record.msg % record.args for record in caplog.records]
        assert sum(["ASGI [1] Started" in message for message in messages]) == 1
        assert sum(["ASGI [1] Send" in message for message in messages]) == 2
        assert sum(["ASGI [1] Receive" in message for message in messages]) == 1
        assert sum(["ASGI [1] Completed" in message for message in messages]) == 1
        assert (
            sum(["ASGI [1] Raised exception" in message for message in messages]) == 0
        )


@pytest.mark.anyio
async def test_message_logger_exc(caplog):
    async def app(scope, receive, send):
        raise RuntimeError()

    with caplog_for_logger(caplog, "uvicorn.asgi"):
        caplog.set_level(TRACE_LOG_LEVEL, logger="uvicorn.asgi")
        caplog.set_level(TRACE_LOG_LEVEL)
        app = MessageLoggerMiddleware(app)
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            with pytest.raises(RuntimeError):
                await client.get("/")
        messages = [record.msg % record.args for record in caplog.records]
        assert sum(["ASGI [1] Started" in message for message in messages]) == 1
        assert sum(["ASGI [1] Send" in message for message in messages]) == 0
        assert sum(["ASGI [1] Receive" in message for message in messages]) == 0
        assert sum(["ASGI [1] Completed" in message for message in messages]) == 0
        assert (
            sum(["ASGI [1] Raised exception" in message for message in messages]) == 1
        )
