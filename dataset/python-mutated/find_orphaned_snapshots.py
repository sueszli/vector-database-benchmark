"""


We want to detect a few things

1. Tests that use the snapshot fixture but don't have a recorded snapshot
2. Snapshots without a corresponding
"""
import pytest
from _pytest.config import Config, PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item

@pytest.hookimpl
def pytest_addoption(parser: Parser, pluginmanager: PytestPluginManager):
    if False:
        i = 10
        return i + 15
    parser.addoption('--filter-fixtures', action='store')

@pytest.hookimpl
def pytest_collection_modifyitems(session: Session, config: Config, items: list[Item]):
    if False:
        for i in range(10):
            print('nop')
    ff = config.getoption('--filter-fixtures')
    if ff:
        filter_fixtures = set(ff.split(','))
        selected = []
        deselected = []
        for item in items:
            if hasattr(item, 'fixturenames') and filter_fixtures.isdisjoint(set(item.fixturenames)):
                deselected.append(item)
            else:
                selected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)