from unittest import mock
from boto3 import client, session
from moto import mock_apigateway
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.apigateway.apigateway_service import Stage
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_apigateway_restapi_client_certificate_enabled:

    def set_mocked_audit_info(self):
        if False:
            print('Hello World!')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_apigateway
    def test_apigateway_no_stages(self):
        if False:
            while True:
                i = 10
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        apigateway_client.create_rest_api(name='test-rest-api')
        from prowler.providers.aws.services.apigateway.apigateway_service import APIGateway
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.apigateway.apigateway_restapi_client_certificate_enabled.apigateway_restapi_client_certificate_enabled.apigateway_client', new=APIGateway(current_audit_info)):
            from prowler.providers.aws.services.apigateway.apigateway_restapi_client_certificate_enabled.apigateway_restapi_client_certificate_enabled import apigateway_restapi_client_certificate_enabled
            check = apigateway_restapi_client_certificate_enabled()
            result = check.execute()
            assert len(result) == 0

    @mock_apigateway
    def test_apigateway_one_stage_without_certificate(self):
        if False:
            return 10
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        rest_api = apigateway_client.create_rest_api(name='test-rest-api')
        root_resource_id = apigateway_client.get_resources(restApiId=rest_api['id'])['items'][0]['id']
        resource = apigateway_client.create_resource(restApiId=rest_api['id'], parentId=root_resource_id, pathPart='test-path')
        apigateway_client.put_method(restApiId=rest_api['id'], resourceId=resource['id'], httpMethod='GET', authorizationType='NONE')
        apigateway_client.put_integration(restApiId=rest_api['id'], resourceId=resource['id'], httpMethod='GET', type='HTTP', integrationHttpMethod='POST', uri='http://test.com')
        apigateway_client.create_deployment(restApiId=rest_api['id'], stageName='test')
        from prowler.providers.aws.services.apigateway.apigateway_service import APIGateway
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.apigateway.apigateway_restapi_client_certificate_enabled.apigateway_restapi_client_certificate_enabled.apigateway_client', new=APIGateway(current_audit_info)):
            from prowler.providers.aws.services.apigateway.apigateway_restapi_client_certificate_enabled.apigateway_restapi_client_certificate_enabled import apigateway_restapi_client_certificate_enabled
            check = apigateway_restapi_client_certificate_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f"API Gateway test-rest-api ID {rest_api['id']} in stage test does not have client certificate enabled."
            assert result[0].resource_id == 'test-rest-api'
            assert result[0].resource_arn == f"arn:{current_audit_info.audited_partition}:apigateway:{AWS_REGION}::/restapis/{rest_api['id']}/stages/test"
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == [None]

    @mock_apigateway
    def test_apigateway_one_stage_with_certificate(self):
        if False:
            for i in range(10):
                print('nop')
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        rest_api = apigateway_client.create_rest_api(name='test-rest-api')
        from prowler.providers.aws.services.apigateway.apigateway_service import APIGateway
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.apigateway.apigateway_restapi_client_certificate_enabled.apigateway_restapi_client_certificate_enabled.apigateway_client', new=APIGateway(current_audit_info)) as service_client:
            from prowler.providers.aws.services.apigateway.apigateway_restapi_client_certificate_enabled.apigateway_restapi_client_certificate_enabled import apigateway_restapi_client_certificate_enabled
            service_client.rest_apis[0].stages.append(Stage(name='test', arn=f'arn:{current_audit_info.audited_partition}:apigateway:{AWS_REGION}::/restapis/test-rest-api/stages/test', logging=True, client_certificate=True, waf=True))
            check = apigateway_restapi_client_certificate_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f"API Gateway test-rest-api ID {rest_api['id']} in stage test has client certificate enabled."
            assert result[0].resource_id == 'test-rest-api'
            assert result[0].resource_arn == f'arn:{current_audit_info.audited_partition}:apigateway:{AWS_REGION}::/restapis/test-rest-api/stages/test'
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []