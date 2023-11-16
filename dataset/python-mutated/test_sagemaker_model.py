from __future__ import annotations
from unittest import mock
from unittest.mock import patch
import pytest
from botocore.exceptions import ClientError
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.sagemaker import SageMakerHook
from airflow.providers.amazon.aws.operators.sagemaker import ApprovalStatus, SageMakerDeleteModelOperator, SageMakerModelOperator, SageMakerRegisterModelVersionOperator
CREATE_MODEL_PARAMS: dict = {'ModelName': 'model_name', 'PrimaryContainer': {'Image': 'image_name', 'ModelDataUrl': 'output_path'}, 'ExecutionRoleArn': 'arn:aws:iam:role/test-role'}
EXPECTED_INTEGER_FIELDS: list[list[str]] = []

class TestSageMakerModelOperator:

    @patch.object(SageMakerHook, 'describe_model', return_value='')
    @patch.object(SageMakerHook, 'create_model')
    def test_execute(self, mock_create_model, _):
        if False:
            while True:
                i = 10
        sagemaker = SageMakerModelOperator(task_id='test_sagemaker_operator', config=CREATE_MODEL_PARAMS)
        mock_create_model.return_value = {'ModelArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 200}}
        sagemaker.execute(None)
        mock_create_model.assert_called_once_with(CREATE_MODEL_PARAMS)
        assert sagemaker.integer_fields == EXPECTED_INTEGER_FIELDS

    @patch.object(SageMakerHook, 'create_model')
    def test_execute_with_failure(self, mock_create_model):
        if False:
            while True:
                i = 10
        sagemaker = SageMakerModelOperator(task_id='test_sagemaker_operator', config=CREATE_MODEL_PARAMS)
        mock_create_model.return_value = {'ModelArn': 'test_arn', 'ResponseMetadata': {'HTTPStatusCode': 404}}
        with pytest.raises(AirflowException):
            sagemaker.execute(None)

class TestSageMakerDeleteModelOperator:

    @patch.object(SageMakerHook, 'delete_model')
    def test_execute(self, delete_model):
        if False:
            return 10
        op = SageMakerDeleteModelOperator(task_id='test_sagemaker_operator', config={'ModelName': 'model_name'})
        op.execute(None)
        delete_model.assert_called_once_with(model_name='model_name')

class TestSageMakerRegisterModelVersionOperator:

    @patch.object(SageMakerHook, 'create_model_package_group')
    @patch('airflow.providers.amazon.aws.hooks.sagemaker.SageMakerHook.conn', new_callable=mock.PropertyMock)
    def test_execute(self, conn_mock, create_group_mock):
        if False:
            for i in range(10):
                print('nop')
        image = '257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.2-1'
        model = 's3://your-bucket-name/model.tar.gz'
        group = 'group-name'
        op = SageMakerRegisterModelVersionOperator(task_id='test', image_uri=image, model_url=model, package_group_name=group, model_approval=ApprovalStatus.APPROVED)
        op.execute(None)
        create_group_mock.assert_called_once_with('group-name', '')
        conn_mock().create_model_package.assert_called_once()
        args_dict = conn_mock().create_model_package.call_args.kwargs
        assert args_dict['InferenceSpecification']['Containers'][0]['Image'] == image
        assert args_dict['InferenceSpecification']['Containers'][0]['ModelDataUrl'] == model
        assert args_dict['ModelPackageGroupName'] == group
        assert args_dict['ModelApprovalStatus'] == 'Approved'

    @pytest.mark.parametrize('group_created', [True, False])
    @patch.object(SageMakerHook, 'create_model_package_group')
    @patch('airflow.providers.amazon.aws.hooks.sagemaker.SageMakerHook.conn', new_callable=mock.PropertyMock)
    def test_group_deleted_if_error_when_adding_model(self, conn_mock, create_group_mock, group_created):
        if False:
            for i in range(10):
                print('nop')
        group = 'group-name'
        op = SageMakerRegisterModelVersionOperator(task_id='test', image_uri='257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.2-1', model_url='s3://your-bucket-name/model.tar.gz', package_group_name=group, model_approval=ApprovalStatus.APPROVED)
        create_group_mock.return_value = group_created
        conn_mock().create_model_package.side_effect = ClientError(error_response={'Error': {'Code': 'ohno'}}, operation_name='empty')
        with pytest.raises(ClientError):
            op.execute(None)
        if group_created:
            conn_mock().delete_model_package_group.assert_called_once_with(ModelPackageGroupName=group)
        else:
            conn_mock().delete_model_package_group.assert_not_called()

    @patch.object(SageMakerHook, 'create_model_package_group')
    @patch('airflow.providers.amazon.aws.hooks.sagemaker.SageMakerHook.conn', new_callable=mock.PropertyMock)
    def test_can_override_parameters_using_extras(self, conn_mock, _):
        if False:
            i = 10
            return i + 15
        response_type = ['test/test']
        op = SageMakerRegisterModelVersionOperator(task_id='test', image_uri='257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.2-1', model_url='s3://your-bucket-name/model.tar.gz', package_group_name='group-name', extras={'InferenceSpecification': {'SupportedResponseMIMETypes': response_type}})
        op.execute(None)
        conn_mock().create_model_package.assert_called_once()
        args_dict = conn_mock().create_model_package.call_args.kwargs
        assert args_dict['InferenceSpecification']['SupportedResponseMIMETypes'] == response_type