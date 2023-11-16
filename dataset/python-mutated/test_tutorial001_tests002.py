import importlib
import pytest
from docs_src.tutorial.fastapi.app_testing.tutorial001 import main as app_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001 import test_main_002 as test_mod

@pytest.fixture(name='prepare', autouse=True)
def prepare_fixture(clear_sqlmodel):
    if False:
        return 10
    importlib.reload(app_mod)
    importlib.reload(test_mod)

def test_tutorial():
    if False:
        i = 10
        return i + 15
    test_mod.test_create_hero()