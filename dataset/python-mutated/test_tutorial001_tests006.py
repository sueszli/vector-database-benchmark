import importlib
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from docs_src.tutorial.fastapi.app_testing.tutorial001 import main as app_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001 import test_main_006 as test_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001.test_main_006 import client_fixture, session_fixture
assert session_fixture, 'This keeps the session fixture used below'
assert client_fixture, 'This keeps the client fixture used below'

@pytest.fixture(name='prepare')
def prepare_fixture(clear_sqlmodel):
    if False:
        for i in range(10):
            print('nop')
    importlib.reload(app_mod)

def test_tutorial(prepare, session: Session, client: TestClient):
    if False:
        return 10
    test_mod.test_create_hero(client)