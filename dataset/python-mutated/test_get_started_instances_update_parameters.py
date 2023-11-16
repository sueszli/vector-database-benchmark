from botocore.exceptions import ClientError
import pytest

class MockManager:

    def __init__(self, stub_runner, instance_data, input_mocker):
        if False:
            while True:
                i = 10
        self.instance_data = instance_data
        self.group_name = 'test-group'
        param_values = [str(ind * 10) for ind in range(1, 4)]
        self.parameters = [{'ParameterName': f'auto_increment_{ind}', 'ParameterValue': param_values[ind - 1], 'IsModifiable': True, 'DataType': 'integer', 'Description': 'Test description', 'AllowedValues': f'{ind}-{ind}00'} for ind in range(1, 4)]
        self.scenario_args = [self.group_name]
        self.scenario_out = None
        input_mocker.mock_answers(param_values)
        self.stub_runner = stub_runner

    def setup_stubs(self, error, stop_on, stubber):
        if False:
            return 10
        with self.stub_runner(error, stop_on) as runner:
            runner.add(stubber.stub_describe_db_parameters, self.group_name, self.parameters)
            runner.add(stubber.stub_modify_db_parameter_group, self.group_name, self.parameters)
            runner.add(stubber.stub_describe_db_parameters, self.group_name, self.parameters, source='user')

@pytest.fixture
def mock_mgr(stub_runner, instance_data, input_mocker):
    if False:
        for i in range(10):
            print('nop')
    return MockManager(stub_runner, instance_data, input_mocker)

def test_update_parameters(mock_mgr, capsys):
    if False:
        print('Hello World!')
    mock_mgr.setup_stubs(None, None, mock_mgr.instance_data.stubber)
    mock_mgr.instance_data.scenario.update_parameters(*mock_mgr.scenario_args)
    capt = capsys.readouterr()
    for param in mock_mgr.parameters:
        assert f"'ParameterName': '{param['ParameterName']}'" in capt.out

@pytest.mark.parametrize('error, stop_on_index', [('TESTERROR-stub_describe_db_parameters', 0), ('TESTERROR-stub_modify_db_parameter_group', 1), ('TESTERROR-stub_describe_db_parameters', 2)])
def test_update_parameters_error(mock_mgr, caplog, error, stop_on_index):
    if False:
        i = 10
        return i + 15
    mock_mgr.setup_stubs(error, stop_on_index, mock_mgr.instance_data.stubber)
    with pytest.raises(ClientError) as exc_info:
        mock_mgr.instance_data.scenario.update_parameters(*mock_mgr.scenario_args)
    assert exc_info.value.response['Error']['Code'] == error
    assert error in caplog.text