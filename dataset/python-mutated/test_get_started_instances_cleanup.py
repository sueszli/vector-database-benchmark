from botocore.exceptions import ClientError
import pytest

class MockManager:

    def __init__(self, stub_runner, instance_data, input_mocker):
        if False:
            print('Hello World!')
        self.instance_data = instance_data
        self.db_inst = {'DBInstanceIdentifier': 'test-instance'}
        self.group = {'DBParameterGroupName': 'test-group'}
        self.scenario_args = [self.db_inst, self.group['DBParameterGroupName']]
        self.scenario_out = None
        answers = ['y']
        input_mocker.mock_answers(answers)
        self.stub_runner = stub_runner

    def setup_stubs(self, error, stop_on, stubber):
        if False:
            for i in range(10):
                print('nop')
        with self.stub_runner(error, stop_on) as runner:
            runner.add(stubber.stub_delete_db_instance, self.db_inst['DBInstanceIdentifier'])
            runner.add(stubber.stub_describe_db_instances, self.db_inst['DBInstanceIdentifier'], error_code='DBInstanceNotFound')
            runner.add(stubber.stub_delete_db_parameter_group, self.group['DBParameterGroupName'])

@pytest.fixture
def mock_mgr(stub_runner, instance_data, input_mocker):
    if False:
        i = 10
        return i + 15
    return MockManager(stub_runner, instance_data, input_mocker)

def test_cleanup(mock_mgr, capsys):
    if False:
        i = 10
        return i + 15
    mock_mgr.setup_stubs(None, None, mock_mgr.instance_data.stubber)
    mock_mgr.instance_data.scenario.cleanup(*mock_mgr.scenario_args)
    capt = capsys.readouterr()
    assert mock_mgr.db_inst['DBInstanceIdentifier'] in capt.out
    assert mock_mgr.group['DBParameterGroupName'] in capt.out

@pytest.mark.parametrize('error, stop_on_index', [('TESTERROR-stub_delete_db_instance', 0), ('TESTERROR-stub_describe_db_instances', 1), ('TESTERROR-stub_delete_db_parameter_group', 2)])
def test_cleanup_error(mock_mgr, caplog, error, stop_on_index):
    if False:
        print('Hello World!')
    mock_mgr.setup_stubs(error, stop_on_index, mock_mgr.instance_data.stubber)
    with pytest.raises(ClientError) as exc_info:
        mock_mgr.instance_data.scenario.cleanup(*mock_mgr.scenario_args)
    assert exc_info.value.response['Error']['Code'] == error
    assert error in caplog.text