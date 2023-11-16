import importlib
import pytest
from sqlmodel import Session
from docs_src.tutorial.fastapi.app_testing.tutorial001 import main as app_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001 import test_main_005 as test_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001.test_main_005 import session_fixture
assert session_fixture, 'This keeps the session fixture used below'

@pytest.fixture(name='prepare')
def prepare_fixture(clear_sqlmodel):
    if False:
        while True:
            i = 10
    importlib.reload(app_mod)

def test_tutorial(prepare, session: Session):
    if False:
        print('Hello World!')
    test_mod.test_create_hero(session)