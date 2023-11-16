import importlib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlalchemy.engine.reflection import Inspector
from sqlmodel import Session, create_engine
from docs_src.tutorial.fastapi.app_testing.tutorial001 import main as app_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001 import test_main as test_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001.test_main import client_fixture, session_fixture
assert session_fixture, 'This keeps the session fixture used below'
assert client_fixture, 'This keeps the client fixture used below'

@pytest.fixture(name='prepare', autouse=True)
def prepare_fixture(clear_sqlmodel):
    if False:
        while True:
            i = 10
    importlib.reload(app_mod)
    importlib.reload(test_mod)

def test_create_hero(session: Session, client: TestClient):
    if False:
        return 10
    test_mod.test_create_hero(client)

def test_create_hero_incomplete(session: Session, client: TestClient):
    if False:
        print('Hello World!')
    test_mod.test_create_hero_incomplete(client)

def test_create_hero_invalid(session: Session, client: TestClient):
    if False:
        for i in range(10):
            print('nop')
    test_mod.test_create_hero_invalid(client)

def test_read_heroes(session: Session, client: TestClient):
    if False:
        return 10
    test_mod.test_read_heroes(session=session, client=client)

def test_read_hero(session: Session, client: TestClient):
    if False:
        i = 10
        return i + 15
    test_mod.test_read_hero(session=session, client=client)

def test_update_hero(session: Session, client: TestClient):
    if False:
        print('Hello World!')
    test_mod.test_update_hero(session=session, client=client)

def test_delete_hero(session: Session, client: TestClient):
    if False:
        i = 10
        return i + 15
    test_mod.test_delete_hero(session=session, client=client)

def test_startup():
    if False:
        while True:
            i = 10
    app_mod.engine = create_engine('sqlite://')
    app_mod.on_startup()
    insp: Inspector = inspect(app_mod.engine)
    assert insp.has_table(str(app_mod.Hero.__tablename__))

def test_get_session():
    if False:
        print('Hello World!')
    app_mod.engine = create_engine('sqlite://')
    for session in app_mod.get_session():
        assert isinstance(session, Session)
        assert session.bind == app_mod.engine

def test_read_hero_not_found(client: TestClient):
    if False:
        while True:
            i = 10
    response = client.get('/heroes/9000')
    assert response.status_code == 404

def test_update_hero_not_found(client: TestClient):
    if False:
        while True:
            i = 10
    response = client.patch('/heroes/9000', json={'name': 'Very-Rusty-Man'})
    assert response.status_code == 404

def test_delete_hero_not_found(client: TestClient):
    if False:
        i = 10
        return i + 15
    response = client.delete('/heroes/9000')
    assert response.status_code == 404