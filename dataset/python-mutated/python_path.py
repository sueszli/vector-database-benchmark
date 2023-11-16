import sys
import pytest
from pytest import Config
from pytest import Parser

def pytest_addoption(parser: Parser) -> None:
    if False:
        return 10
    parser.addini('pythonpath', type='paths', help='Add paths to sys.path', default=[])

@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config: Config) -> None:
    if False:
        i = 10
        return i + 15
    for path in reversed(early_config.getini('pythonpath')):
        sys.path.insert(0, str(path))

@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: Config) -> None:
    if False:
        while True:
            i = 10
    for path in config.getini('pythonpath'):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)