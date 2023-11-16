from __future__ import annotations
from unittest import mock
import pytest
from botocore.exceptions import ClientError
from airflow.exceptions import AirflowException, TaskDeferred
from airflow.providers.amazon.aws.hooks.sagemaker import SageMakerHook
from airflow.providers.amazon.aws.operators import sagemaker
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerEndpointOperator
from airflow.providers.amazon.aws.triggers.sagemaker import SageMakerTrigger
CREATE_MODEL_PARAMS: dict = {'ModelName': 'model_name', 'PrimaryContainer': {'Image': 'image_name', 'ModelDataUrl': 'output_path'}, 'ExecutionRoleArn': 'arn:aws:iam:role/test-role'}
CREATE_ENDPOINT_CONFIG_PARAMS: dict = {'EndpointConfigName': 'config_name', 'ProductionVariants': [{'VariantName': 'AllTraffic', 'ModelName': 'model_name', 'InitialInstanceCount': '1', 'InstanceType': 'ml.c4.xlarge'}]}
CREATE_ENDPOINT_PARAMS: dict = {'EndpointName': 'endpoint_name', 'EndpointConfigName': 'config_name'}
CONFIG: dict = {'Model': CREATE_MODEL_PARAMS, 'EndpointConfig': CREATE_ENDPOINT_CONFIG_PARAMS, 'Endpoint': CREATE_ENDPOINT_PARAMS}
EXPECTED_INTEGER_FIELDS: list[list[str]] = [['EndpointConfig', 'ProductionVariants', 'InitialInstanceCount']]

class TestSageMakerEndpointOperator:

    def setup_method(self):
        if False:
            return 10
        self.sagemaker = SageMakerEndpointOperator(task_id='test_sagemaker_operator', config=CONFIG, wait_for_completion=False, check_interval=5, operation='create')

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_model')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(SageMakerHook, 'create_endpoint')
    @mock.patch.object(sagemaker, 'serialize', return_value='')
    def test_integer_fields(self, serialize, mock_endpoint, mock_endpoint_config, mock_model, mock_client):
        if False:
            return 10
        mock_endpoint.return_value = {'EndpointArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        self.sagemaker.execute(None)
        assert self.sagemaker.integer_fields == EXPECTED_INTEGER_FIELDS
        for variant in self.sagemaker.config['EndpointConfig']['ProductionVariants']:
            assert variant['InitialInstanceCount'] == int(variant['InitialInstanceCount'])

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_model')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(SageMakerHook, 'create_endpoint')
    @mock.patch.object(sagemaker, 'serialize', return_value='')
    def test_execute(self, serialize, mock_endpoint, mock_endpoint_config, mock_model, mock_client):
        if False:
            i = 10
            return i + 15
        mock_endpoint.return_value = {'EndpointArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        self.sagemaker.execute(None)
        mock_model.assert_called_once_with(CREATE_MODEL_PARAMS)
        mock_endpoint_config.assert_called_once_with(CREATE_ENDPOINT_CONFIG_PARAMS)
        mock_endpoint.assert_called_once_with(CREATE_ENDPOINT_PARAMS, wait_for_completion=False)
        assert self.sagemaker.integer_fields == EXPECTED_INTEGER_FIELDS
        for variant in self.sagemaker.config['EndpointConfig']['ProductionVariants']:
            assert variant['InitialInstanceCount'] == int(variant['InitialInstanceCount'])

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_model')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(SageMakerHook, 'create_endpoint')
    def test_execute_with_failure(self, mock_endpoint, mock_endpoint_config, mock_model, mock_client):
        if False:
            i = 10
            return i + 15
        mock_endpoint.return_value = {'EndpointArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 404}}
        with pytest.raises(AirflowException):
            self.sagemaker.execute(None)

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_model')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(SageMakerHook, 'create_endpoint')
    @mock.patch.object(SageMakerHook, 'update_endpoint')
    @mock.patch.object(sagemaker, 'serialize', return_value='')
    def test_execute_with_duplicate_endpoint_creation(self, serialize, mock_endpoint_update, mock_endpoint, mock_endpoint_config, mock_model, mock_client):
        if False:
            for i in range(10):
                print('nop')
        response = {'Error': {'Code': 'ValidationException', 'Message': 'Cannot create already existing endpoint.'}}
        mock_endpoint.side_effect = ClientError(error_response=response, operation_name='CreateEndpoint')
        mock_endpoint_update.return_value = {'EndpointArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        self.sagemaker.execute(None)

    @mock.patch.object(SageMakerHook, 'get_conn')
    @mock.patch.object(SageMakerHook, 'create_model')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(SageMakerHook, 'create_endpoint')
    @mock.patch.object(SageMakerHook, 'update_endpoint')
    @mock.patch.object(sagemaker, 'serialize', return_value='')
    def test_execute_with_duplicate_endpoint_removes_tags(self, serialize, mock_endpoint_update, mock_endpoint_create, mock_endpoint_config, mock_model, mock_client):
        if False:
            i = 10
            return i + 15
        mock_endpoint_create.side_effect = ClientError(error_response={'Error': {'Code': 'ValidationException', 'Message': 'Cannot create already existing endpoint.'}}, operation_name='CreateEndpoint')

        def _check_no_tags(config, wait_for_completion):
            if False:
                return 10
            assert 'Tags' not in config
            return {'EndpointArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        mock_endpoint_update.side_effect = _check_no_tags
        self.sagemaker.config['Endpoint']['Tags'] = {'Key': 'k', 'Value': 'v'}
        self.sagemaker.execute(None)

    @mock.patch.object(SageMakerHook, 'create_model')
    @mock.patch.object(SageMakerHook, 'create_endpoint_config')
    @mock.patch.object(SageMakerHook, 'create_endpoint')
    def test_deferred(self, mock_create_endpoint, _, __):
        if False:
            while True:
                i = 10
        self.sagemaker.deferrable = True
        mock_create_endpoint.return_value = {'ResponseMetadata': {'HTTPStatusCode': 200}}
        with pytest.raises(TaskDeferred) as defer:
            self.sagemaker.execute(None)
        assert isinstance(defer.value.trigger, SageMakerTrigger)
        assert defer.value.trigger.job_name == 'endpoint_name'
        assert defer.value.trigger.job_type == 'endpoint'