import time
import pytest
from aurora_wrapper import AuroraWrapper
import scenario_get_started_aurora

@pytest.fixture(autouse=True)
def mock_wait(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(scenario_get_started_aurora, 'wait', lambda x: time.sleep(x * 5))

@pytest.mark.integ
def test_run_cluster_scenario_integ(input_mocker, capsys):
    if False:
        i = 10
        return i + 15
    scenario = scenario_get_started_aurora.AuroraClusterScenario(AuroraWrapper.from_client())
    input_mocker.mock_answers([1, '1', '1', 'admin', 'password', 1, 1, 'y', 'y'])
    scenario.run_scenario('aurora-mysql', 'doc-example-test-cluster-group', 'doc-example-test-aurora', 'docexampletestdb')
    capt = capsys.readouterr()
    assert 'Thanks for watching!' in capt.out