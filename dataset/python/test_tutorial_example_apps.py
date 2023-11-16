from __future__ import annotations

import sys

import pytest
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from docs.examples.contrib.sqlalchemy.plugins.tutorial import (
    full_app_no_plugins,
    full_app_with_init_plugin,
    full_app_with_plugin,
    full_app_with_serialization_plugin,
    full_app_with_session_di,
)
from sqlalchemy.ext.asyncio import create_async_engine

from litestar import Litestar
from litestar.testing import TestClient


@pytest.fixture(
    params=[
        full_app_no_plugins,
        full_app_with_init_plugin,
        full_app_with_plugin,
        full_app_with_serialization_plugin,
        full_app_with_session_di,
    ]
)
async def app(monkeypatch: MonkeyPatch, request: FixtureRequest) -> Litestar:
    from docs.examples.contrib.sqlalchemy.plugins.tutorial.full_app_no_plugins import Base

    app_module = request.param

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        monkeypatch.setattr(app_module, "create_async_engine", lambda *a, **kw: engine)
    except AttributeError:
        app_module.db_config.connection_string = None
        app_module.db_config.engine_instance = engine

    return app_module.app


@pytest.mark.skipif(sys.platform != "linux", reason="Unknown - fails on Windows and macOS, in CI only")
def test_no_plugins_full_app(app: Litestar) -> None:
    todo = {"title": "Start writing todo list", "done": True}
    todo_list = [todo]

    with TestClient(app) as client:
        response = client.post("/", json=todo)
        assert response.status_code == 201
        assert response.json() == todo

        response = client.post("/", json=todo)
        assert response.status_code == 409

        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == todo_list

        response = client.get("/?done=false")
        assert response.status_code == 200
        assert response.json() == []

        response = client.put("/Start writing another list", json=todo)
        assert response.status_code == 404

        updated_todo = dict(todo)
        updated_todo["done"] = False
        response = client.put("/Start writing todo list", json=updated_todo)
        assert response.status_code == 200
        assert response.json() == updated_todo
