import importlib
import pytest
from docs_src.tutorial.fastapi.app_testing.tutorial001 import main as app_mod
from docs_src.tutorial.fastapi.app_testing.tutorial001 import test_main_003 as test_mod

@pytest.fixture(name='prepare', autouse=True)
def prepare_fixture(clear_sqlmodel):
    if False:
        for i in range(10):
            print('nop')
    importlib.reload(app_mod)
    importlib.reload(test_mod)

def test_tutorial():
    if False:
        print('Hello World!')
    test_mod.test_create_hero()