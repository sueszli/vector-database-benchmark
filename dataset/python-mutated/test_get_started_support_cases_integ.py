import boto3
import pytest
from support_wrapper import SupportWrapper
from get_started_support_cases import SupportCasesScenario

@pytest.fixture
def mock_wait(monkeypatch):
    if False:
        print('Hello World!')
    return

@pytest.mark.integ
def test_run_get_started_scenario_integ(input_mocker, capsys):
    if False:
        return 10
    support_client = boto3.client('support')
    support_wrapper = SupportWrapper(support_client)
    scenario = SupportCasesScenario(support_wrapper)
    input_mocker.mock_answers([1, 1, 1])
    scenario.run_scenario()
    capt = capsys.readouterr()
    assert 'Thanks for watching!' in capt.out