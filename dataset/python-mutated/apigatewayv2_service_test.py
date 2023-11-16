import botocore
from boto3 import client, session
from mock import patch
from moto import mock_apigatewayv2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.apigatewayv2.apigatewayv2_service import ApiGatewayV2
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        for i in range(10):
            print('nop')
    '\n    We have to mock every AWS API call using Boto3\n\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n    '
    if operation_name == 'GetAuthorizers':
        return {'Items': [{'AuthorizerId': 'authorizer-id', 'Name': 'test-authorizer'}]}
    elif operation_name == 'GetStages':
        return {'Items': [{'AccessLogSettings': {'DestinationArn': 'string', 'Format': 'string'}, 'StageName': 'test-stage'}]}
    return make_api_call(self, operation_name, kwarg)

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
class Test_ApiGatewayV2_Service:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_apigatewayv2
    def test_service(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        assert apigatewayv2.service == 'apigatewayv2'

    @mock_apigatewayv2
    def test_client(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        for regional_client in apigatewayv2.regional_clients.values():
            assert regional_client.__class__.__name__ == 'ApiGatewayV2'

    @mock_apigatewayv2
    def test__get_session__(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        assert apigatewayv2.session.__class__.__name__ == 'Session'

    @mock_apigatewayv2
    def test_audited_account(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        assert apigatewayv2.audited_account == AWS_ACCOUNT_NUMBER

    @mock_apigatewayv2
    def test__get_apis__(self):
        if False:
            for i in range(10):
                print('nop')
        apigatewayv2_client = client('apigatewayv2', region_name=AWS_REGION)
        apigatewayv2_client.create_api(Name='test-api', ProtocolType='HTTP', Tags={'test': 'test'})
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        assert len(apigatewayv2.apis) == len(apigatewayv2_client.get_apis()['Items'])
        assert apigatewayv2.apis[0].tags == [{'test': 'test'}]

    @mock_apigatewayv2
    def test__get_authorizers__(self):
        if False:
            while True:
                i = 10
        apigatewayv2_client = client('apigatewayv2', region_name=AWS_REGION)
        api = apigatewayv2_client.create_api(Name='test-api', ProtocolType='HTTP')
        apigatewayv2_client.create_authorizer(ApiId=api['ApiId'], AuthorizerType='REQUEST', IdentitySource=[], Name='auth1', AuthorizerPayloadFormatVersion='2.0')
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        assert apigatewayv2.apis[0].authorizer is True

    @mock_apigatewayv2
    def test__get_stages__(self):
        if False:
            for i in range(10):
                print('nop')
        apigatewayv2_client = client('apigatewayv2', region_name=AWS_REGION)
        apigatewayv2_client.create_api(Name='test-api', ProtocolType='HTTP')
        audit_info = self.set_mocked_audit_info()
        apigatewayv2 = ApiGatewayV2(audit_info)
        assert apigatewayv2.apis[0].stages[0].logging is True