from boto3 import client, session
from moto import mock_apigateway
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.apigateway.apigateway_service import APIGateway
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'

class Test_APIGateway_Service:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_apigateway
    def test_service(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert apigateway.service == 'apigateway'

    @mock_apigateway
    def test_client(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        for regional_client in apigateway.regional_clients.values():
            assert regional_client.__class__.__name__ == 'APIGateway'

    @mock_apigateway
    def test__get_session__(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert apigateway.session.__class__.__name__ == 'Session'

    @mock_apigateway
    def test_audited_account(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert apigateway.audited_account == AWS_ACCOUNT_NUMBER

    @mock_apigateway
    def test__get_rest_apis__(self):
        if False:
            while True:
                i = 10
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        apigateway_client.create_rest_api(name='test-rest-api')
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert len(apigateway.rest_apis) == len(apigateway_client.get_rest_apis()['items'])

    @mock_apigateway
    def test__get_authorizers__(self):
        if False:
            print('Hello World!')
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        rest_api = apigateway_client.create_rest_api(name='test-rest-api')
        apigateway_client.create_authorizer(name='test-authorizer', restApiId=rest_api['id'], type='TOKEN')
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert apigateway.rest_apis[0].authorizer is True

    @mock_apigateway
    def test__get_rest_api__(self):
        if False:
            i = 10
            return i + 15
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        apigateway_client.create_rest_api(name='test-rest-api', endpointConfiguration={'types': ['PRIVATE']}, tags={'test': 'test'})
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert apigateway.rest_apis[0].public_endpoint is False
        assert apigateway.rest_apis[0].tags == [{'test': 'test'}]

    @mock_apigateway
    def test__get_stages__(self):
        if False:
            for i in range(10):
                print('nop')
        apigateway_client = client('apigateway', region_name=AWS_REGION)
        rest_api = apigateway_client.create_rest_api(name='test-rest-api')
        root_resource_id = apigateway_client.get_resources(restApiId=rest_api['id'])['items'][0]['id']
        resource = apigateway_client.create_resource(restApiId=rest_api['id'], parentId=root_resource_id, pathPart='test-path')
        apigateway_client.put_method(restApiId=rest_api['id'], resourceId=resource['id'], httpMethod='GET', authorizationType='NONE')
        apigateway_client.put_integration(restApiId=rest_api['id'], resourceId=resource['id'], httpMethod='GET', type='HTTP', integrationHttpMethod='POST', uri='http://test.com')
        apigateway_client.create_deployment(restApiId=rest_api['id'], stageName='test')
        apigateway_client.update_stage(restApiId=rest_api['id'], stageName='test', patchOperations=[{'op': 'replace', 'path': '/*/*/logging/loglevel', 'value': 'INFO'}])
        audit_info = self.set_mocked_audit_info()
        apigateway = APIGateway(audit_info)
        assert apigateway.rest_apis[0].stages[0].logging is True