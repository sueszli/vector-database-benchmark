from botocore.exceptions import ClientError
import pytest

class MockManager:

    def __init__(self, stub_runner, scenario_data, input_mocker):
        if False:
            return 10
        self.scenario_data = scenario_data
        self.service = {'name': 'test-service', 'code': 'service-free-tier'}
        self.category = {'name': 'test-category', 'code': 'why-was-i-billed'}
        self.severity = {'name': 'test-severity', 'code': 'low'}
        self.case_id = 'test-case-id'
        self.scenario_args = [self.service, self.category, self.severity]
        answers = []
        input_mocker.mock_answers(answers)
        self.stub_runner = stub_runner

    def setup_stubs(self, error, stop_on, stubber):
        if False:
            return 10
        with self.stub_runner(error, stop_on) as runner:
            runner.add(stubber.stub_create_case, self.service, self.category, self.severity, self.case_id)

@pytest.fixture
def mock_mgr(stub_runner, scenario_data, input_mocker):
    if False:
        return 10
    return MockManager(stub_runner, scenario_data, input_mocker)

def test_create_case(mock_mgr, capsys):
    if False:
        i = 10
        return i + 15
    mock_mgr.setup_stubs(None, None, mock_mgr.scenario_data.stubber)
    mock_mgr.scenario_data.scenario.create_example_case(*mock_mgr.scenario_args)
    capt = capsys.readouterr()
    assert mock_mgr.case_id in capt.out

@pytest.mark.parametrize('error, stop_on_index', [('TESTERROR-stub_create_example_case', 0)])
def test_cleanup_error(mock_mgr, caplog, error, stop_on_index):
    if False:
        return 10
    mock_mgr.setup_stubs(error, stop_on_index, mock_mgr.scenario_data.stubber)
    with pytest.raises(ClientError) as exc_info:
        mock_mgr.scenario_data.scenario.create_example_case(*mock_mgr.scenario_args)
    assert exc_info.value.response['Error']['Code'] == error
    assert error in caplog.text