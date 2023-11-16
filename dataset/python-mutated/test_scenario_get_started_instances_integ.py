import boto3
import pytest
from elastic_ip import ElasticIpWrapper
from instance import InstanceWrapper
from key_pair import KeyPairWrapper
from security_group import SecurityGroupWrapper
from scenario_get_started_instances import Ec2InstanceScenario

@pytest.mark.integ
def test_run_cluster_scenario_integ(input_mocker, capsys):
    if False:
        return 10
    scenario = Ec2InstanceScenario(InstanceWrapper.from_resource(), KeyPairWrapper.from_resource(), SecurityGroupWrapper.from_resource(), ElasticIpWrapper.from_resource(), boto3.client('ssm'))
    input_mocker.mock_answers(['doc-example-test-key', 'y', 'doc-example-test-group', '', 1, 1, '', '', '', '', 'y'])
    scenario.run_scenario()
    capt = capsys.readouterr()
    assert 'Thanks for watching!' in capt.out