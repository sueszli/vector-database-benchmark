import pytest
import requests
from tests.integration.local.start_api.start_api_integ_base import StartApiIntegBaseClass, WritableStartApiIntegBaseClass
from parameterized import parameterized_class

@parameterized_class(('parameter_overrides', 'template_path'), [({'AuthOverride': 'RequestAuthorizerV2'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthOverride': 'RequestAuthorizerV2Simple'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthOverride': 'RequestAuthorizerV1'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthOverride': 'Token'}, '/testdata/start_api/lambda_authorizers/serverless-api-props.yaml'), ({'AuthOverride': 'Request'}, '/testdata/start_api/lambda_authorizers/serverless-api-props.yaml')])
class TestSfnPropertiesLambdaAuthorizers(StartApiIntegBaseClass):

    def setUp(self):
        if False:
            return 10
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_invokes_authorizer(self):
        if False:
            print('Hello World!')
        headers = {'header': 'myheader'}
        query = {'query': 'myquery'}
        response = requests.get(f'{self.url}/requestauthorizer', headers=headers, params=query, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json, {'message': 'from authorizer'})

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_missing_identity_sources(self):
        if False:
            return 10
        response = requests.get(f'{self.url}/requestauthorizer', timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response_json, {'message': 'Unauthorized'})

@parameterized_class(('parameter_overrides', 'template_path'), [({'AuthHandler': 'app.unauth', 'AuthOverride': 'RequestAuthorizerV1'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthSimpleHandler': 'app.unauthv2', 'AuthOverride': 'RequestAuthorizerV2Simple'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthHandler': 'app.unauth', 'AuthOverride': 'Token'}, '/testdata/start_api/lambda_authorizers/serverless-api-props.yaml'), ({'AuthHandler': 'app.unauth', 'AuthOverride': 'Request'}, '/testdata/start_api/lambda_authorizers/serverless-api-props.yaml')])
class TestSfnPropertiesLambdaAuthorizersUnauthorized(StartApiIntegBaseClass):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_unauthorized_request(self):
        if False:
            print('Hello World!')
        headers = {'header': 'myheader'}
        query = {'query': 'myquery'}
        response = requests.get(f'{self.url}/requestauthorizer', headers=headers, params=query, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response_json, {'message': 'User is not authorized to access this resource'})

@parameterized_class(('parameter_overrides', 'template_path'), [({'AuthHandler': 'app.throws_exception', 'AuthOverride': 'RequestAuthorizerV1'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthSimpleHandler': 'app.throws_exception', 'AuthOverride': 'RequestAuthorizerV2Simple'}, '/testdata/start_api/lambda_authorizers/serverless-http-props.yaml'), ({'AuthHandler': 'app.throws_exception', 'AuthOverride': 'Token'}, '/testdata/start_api/lambda_authorizers/serverless-api-props.yaml'), ({'AuthHandler': 'app.throws_exception', 'AuthOverride': 'Request'}, '/testdata/start_api/lambda_authorizers/serverless-api-props.yaml')])
class TestSfnPropertiesLambdaAuthorizer500(StartApiIntegBaseClass):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_unauthorized_request(self):
        if False:
            while True:
                i = 10
        headers = {'header': 'myheader'}
        query = {'query': 'myquery'}
        response = requests.get(f'{self.url}/requestauthorizer', headers=headers, params=query, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response_json, {'message': 'Internal server error'})

class TestUsingSimpleResponseWithV1HttpApi(WritableStartApiIntegBaseClass):
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  TestServerlessHttpApi:\n    Type: AWS::Serverless::HttpApi\n    Properties:\n      StageName: http\n      Auth:\n        DefaultAuthorizer: RequestAuthorizerV2\n        Authorizers:\n          RequestAuthorizerV2:\n            AuthorizerPayloadFormatVersion: "1.0"\n            EnableSimpleResponses: true\n            FunctionArn: !GetAtt AuthorizerFunction.Arn\n            Identity:\n              Headers:\n                - header\n              QueryStrings:\n                - query\n  HelloWorldFunction:\n    Type: AWS::Serverless::Function\n    Properties:\n      CodeUri: ./\n      Handler: app.lambda_handler\n      Events:\n        ApiEvent:\n          Type: HttpApi\n          Properties:\n            Path: /requestauthorizer\n            Method: get\n            ApiId: !Ref TestServerlessHttpApi\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            return 10
        self.assertIn("EnableSimpleResponses must be used with the 2.0 payload format version in Lambda Authorizer 'RequestAuthorizerV2'.", self.start_api_process_output)

class TestInvalidInvalidVersionHttpApi(WritableStartApiIntegBaseClass):
    """
    Test using an invalid AuthorizerPayloadFormatVersion property value
    when defining a Lambda Authorizer in the a Serverless resource properties.
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  TestServerlessHttpApi:\n    Type: AWS::Serverless::HttpApi\n    Properties:\n      StageName: http\n      Auth:\n        DefaultAuthorizer: RequestAuthorizerV2\n        Authorizers:\n          RequestAuthorizerV2:\n            AuthorizerPayloadFormatVersion: "3.0"\n            EnableSimpleResponses: false\n            FunctionArn: !GetAtt AuthorizerFunction.Arn\n            Identity:\n              Headers:\n                - header\n              QueryStrings:\n                - query\n  HelloWorldFunction:\n    Type: AWS::Serverless::Function\n    Properties:\n      CodeUri: ./\n      Handler: app.lambda_handler\n      Events:\n        ApiEvent:\n          Type: HttpApi\n          Properties:\n            Path: /requestauthorizer\n            Method: get\n            ApiId: !Ref TestServerlessHttpApi\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            i = 10
            return i + 15
        self.assertIn("Error: Lambda Authorizer 'RequestAuthorizerV2' must contain a valid 'AuthorizerPayloadFormatVersion' for HTTP APIs.", self.start_api_process_output)

class TestUsingInvalidFunctionArnHttpApi(WritableStartApiIntegBaseClass):
    """
    Test using an invalid FunctionArn property value when defining
    a Lambda Authorizer in the a Serverless resource properties.
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  TestServerlessHttpApi:\n    Type: AWS::Serverless::HttpApi\n    Properties:\n      StageName: http\n      Auth:\n        DefaultAuthorizer: RequestAuthorizerV2\n        Authorizers:\n          RequestAuthorizerV2:\n            AuthorizerPayloadFormatVersion: "2.0"\n            EnableSimpleResponses: false\n            FunctionArn: iofaqio\'hfw;iqauh\n            Identity:\n              Headers:\n                - header\n              QueryStrings:\n                - query\n  HelloWorldFunction:\n    Type: AWS::Serverless::Function\n    Properties:\n      CodeUri: ./\n      Handler: app.lambda_handler\n      Events:\n        ApiEvent:\n          Type: HttpApi\n          Properties:\n            Path: /requestauthorizer\n            Method: get\n            ApiId: !Ref TestServerlessHttpApi\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            i = 10
            return i + 15
        self.assertIn("Unable to parse the Lambda ARN for Authorizer 'RequestAuthorizerV2', skipping", self.start_api_process_output)