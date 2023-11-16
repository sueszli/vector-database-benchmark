import os
import pytest
from keyspace import KeyspaceWrapper
from scenario_get_started_keyspaces import KeyspaceScenario
from query import QueryManager

@pytest.fixture
def mock_wait(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    return

@pytest.mark.skip(reason='Skip until shared resources are part of the Docker environment.')
@pytest.mark.integ
def test_run_keyspace_scenario_integ(input_mocker, capsys):
    if False:
        for i in range(10):
            print('nop')
    scenario = KeyspaceScenario(KeyspaceWrapper.from_client())
    input_mocker.mock_answers(['doc_example_test_keyspace', 'movietabletest', '', 1, '', 'y', 'y'])
    scenario.run_scenario()
    capt = capsys.readouterr()
    assert 'Thanks for watching!' in capt.out