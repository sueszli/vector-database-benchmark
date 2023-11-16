from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.sagemaker import SageMakerHook
from airflow.providers.amazon.aws.operators import sagemaker
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerEndpointConfigOperator
CREATE_ENDPOINT_CONFIG_PARAMS: dict = {'EndpointConfigName': 'config_name', 'ProductionVariants': [{'VariantName': 'AllTraffic', 'ModelName': 'model_name', 'InitialInstanceCount': '1', 'InstanceType': 'ml.c4.xlarge'}]}
EXPECTED_INTEGER_FIELDS: list[list[str]] = [['ProductionVariants', 'InitialInstanceCount']]

class TestSageMakerEndpointConfigOperator:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.sagemaker = SageMakerEndpointConfigOperator(task_id='test_sagemaker_operator', config=CREATE_ENDPOINT_CONFIG_PARAMS)

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(sagemaker, 'serialize', return_value='')
    def test_integer_fields(self, serialize, mock_model, mock_client):
        if False:
            print('Hello World!')
        mock_model.return_value = {'EndpointConfigArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        self.sagemaker.execute(None)
        assert self.sagemaker.integer_fields == EXPECTED_INTEGER_FIELDS
        for variant in self.sagemaker.config['ProductionVariants']:
            assert variant['InitialInstanceCount'] == int(variant['InitialInstanceCount'])

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(sagemaker, 'serialize', return_value='')
    def test_execute(self, serialize, mock_model, mock_client):
        if False:
            i = 10
            return i + 15
        mock_model.return_value = {'EndpointConfigArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        self.sagemaker.execute(None)
        mock_model.assert_called_once_with(CREATE_ENDPOINT_CONFIG_PARAMS)

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_model')
    def test_execute_with_failure(self, mock_model, mock_client):
        if False:
            while True:
                i = 10
        mock_model.return_value = {'EndpointConfigArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        with pytest.raises(AirflowException):
            self.sagemaker.execute(None)