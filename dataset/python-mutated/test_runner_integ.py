import logging
import pytest
from auto_scaler import AutoScaler
from load_balancer import LoadBalancer
from parameters import ParameterHelper
from recommendation_service import RecommendationService
import runner

@pytest.mark.integ
def test_runner_integ(input_mocker, caplog):
    if False:
        print('Hello World!')
    caplog.set_level(logging.INFO)
    prefix = 'doc-example-test-resilience'
    table_name = 'doc-example-test-recommendation-service'
    recommendation = RecommendationService.from_client(table_name)
    autoscaler = AutoScaler.from_client(prefix)
    loadbalancer = LoadBalancer.from_client(prefix)
    param_helper = ParameterHelper.from_client(recommendation.table_name)
    scenario = runner.Runner('test/resources', recommendation, autoscaler, loadbalancer, param_helper)
    input_mocker.mock_answers(['', '', 'y', '', '1', '2', '3', '3', '3', '3', '3', '3', '3', 'y'])
    scenario.deploy()
    scenario.demo()
    scenario.destroy()
    assert f'Table {table_name} deleted.' in caplog.text