import pytest
import requests
from tests.integration.local.start_api.start_api_integ_base import StartApiIntegBaseClass, WritableStartApiIntegBaseClass
from parameterized import parameterized_class

@parameterized_class(('template_path', 'endpoint', 'parameter_overrides'), [('/testdata/start_api/lambda_authorizers/cfn-apigw-v1.yaml', 'requestauthorizertoken', {}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v1.yaml', 'requestauthorizerrequest', {}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2', {'RoutePayloadFormatVersion': '2.0'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2', {'RoutePayloadFormatVersion': '1.0'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2simple', {'AuthHandler': 'app.simple_handler', 'RoutePayloadFormatVersion': '2.0'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2simple', {'AuthHandler': 'app.simple_handler', 'RoutePayloadFormatVersion': '1.0'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv1', {'RoutePayloadFormatVersion': '2.0'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv1', {'RoutePayloadFormatVersion': '1.0'})])
class TestCfnLambdaAuthorizerResources(StartApiIntegBaseClass):

    def setUp(self):
        if False:
            return 10
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_invokes_authorizer(self):
        if False:
            return 10
        headers = {'header': 'myheader'}
        query_string = {'query': 'myquery'}
        response = requests.get(f'{self.url}/{self.endpoint}', headers=headers, params=query_string, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json, {'message': 'from authorizer'})

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_missing_identity_sources(self):
        if False:
            return 10
        response = requests.get(f'{self.url}/{self.endpoint}', timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response_json, {'message': 'Unauthorized'})

@parameterized_class(('template_path', 'endpoint', 'parameter_overrides'), [('/testdata/start_api/lambda_authorizers/cfn-apigw-v1.yaml', 'requestauthorizertoken', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v1.yaml', 'requestauthorizerrequest', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2simple', {'AuthHandler': 'app.unauthv2'}), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv1', {'AuthHandler': 'app.unauth'})])
class TestCfnLambdaAuthorizersUnauthorized(StartApiIntegBaseClass):

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
        query_string = {'query': 'myquery'}
        response = requests.get(f'{self.url}/{self.endpoint}', headers=headers, params=query_string, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response_json, {'message': 'User is not authorized to access this resource'})

@parameterized_class(('template_path', 'endpoint'), [('/testdata/start_api/lambda_authorizers/cfn-apigw-v1.yaml', 'requestauthorizertoken'), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v1.yaml', 'requestauthorizerrequest'), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2'), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv2simple'), ('/testdata/start_api/lambda_authorizers/cfn-apigw-v2.yaml', 'requestauthorizerv1')])
class TestCfnLambdaAuthorizer500(StartApiIntegBaseClass):
    parameter_overrides = {'AuthHandler': 'app.throws_exception'}

    def setUp(self):
        if False:
            while True:
                i = 10
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_authorizer_raises_exception(self):
        if False:
            i = 10
            return i + 15
        headers = {'header': 'myheader'}
        query_string = {'query': 'myquery'}
        response = requests.get(f'{self.url}/{self.endpoint}', headers=headers, params=query_string, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response_json, {'message': 'Internal server error'})

class TestInvalidApiTemplateUsingUnsupportedType(WritableStartApiIntegBaseClass):
    """
    Test using an invalid Type for an Authorizer
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  RequestAuthorizer:\n    Type: AWS::ApiGateway::Authorizer\n    Properties:\n      AuthorizerUri: arn:aws:apigateway:123:lambda:path/2015-03-31/functions/arn/invocations\n      Type: notvalid\n      IdentitySource: "method.request.header.header, method.request.querystring.query"\n      Name: RequestAuthorizer\n      RestApiId: !Ref RestApiLambdaAuth\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            return 10
        self.assertIn("Authorizer 'RequestAuthorizer' with type 'notvalid' is currently not supported. Only Lambda Authorizers of type TOKEN and REQUEST are supported.", self.start_api_process_output)

class TestInvalidHttpTemplateUsingIncorrectPayloadVersion(WritableStartApiIntegBaseClass):
    """
    Test using an invalid AuthorizerPayloadFormatVersion for an Authorizer
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  RequestAuthorizerV2Simple:\n    Type: AWS::ApiGatewayV2::Authorizer\n    Properties:\n      AuthorizerPayloadFormatVersion: "3.0"\n      EnableSimpleResponses: false\n      AuthorizerType: REQUEST\n      AuthorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n      IdentitySource:\n        - "$request.header.header"\n        - "$request.querystring.query"\n      Name: RequestAuthorizerV2Simple\n      ApiId: !Ref HttpLambdaAuth\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            while True:
                i = 10
        self.assertIn("Error: Lambda Authorizer 'RequestAuthorizerV2Simple' contains an invalid 'AuthorizerPayloadFormatVersion', it must be set to '1.0' or '2.0'", self.start_api_process_output)

class TestInvalidHttpTemplateSimpleResponseWithV1(WritableStartApiIntegBaseClass):
    """
    Test using simple responses with V1 format version
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  RequestAuthorizerV2Simple:\n    Type: AWS::ApiGatewayV2::Authorizer\n    Properties:\n      AuthorizerPayloadFormatVersion: "1.0"\n      EnableSimpleResponses: true\n      AuthorizerType: REQUEST\n      AuthorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n      IdentitySource:\n        - "$request.header.header"\n        - "$request.querystring.query"\n      Name: RequestAuthorizerV2Simple\n      ApiId: !Ref HttpLambdaAuth\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            while True:
                i = 10
        self.assertIn("Error: 'EnableSimpleResponses' is only supported for '2.0' payload format versions for Lambda Authorizer 'RequestAuthorizerV2Simple'.", self.start_api_process_output)

class TestInvalidHttpTemplateUnsupportedType(WritableStartApiIntegBaseClass):
    """
    Test using an invalid Type for HttpApi
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  RequestAuthorizerV2Simple:\n    Type: AWS::ApiGatewayV2::Authorizer\n    Properties:\n      AuthorizerPayloadFormatVersion: "1.0"\n      EnableSimpleResponses: false\n      AuthorizerType: unsupportedtype\n      AuthorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n      IdentitySource:\n        - "$request.header.header"\n        - "$request.querystring.query"\n      Name: RequestAuthorizerV2Simple\n      ApiId: !Ref HttpLambdaAuth\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            while True:
                i = 10
        self.assertIn("Authorizer 'RequestAuthorizerV2Simple' with type 'unsupportedtype' is currently not supported. Only Lambda Authorizers of type REQUEST are supported for API Gateway V2.", self.start_api_process_output)

class TestInvalidHttpTemplateInvalidIdentitySources(WritableStartApiIntegBaseClass):
    """
    Test using an invalid identity source
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  RequestAuthorizerV2Simple:\n    Type: AWS::ApiGatewayV2::Authorizer\n    Properties:\n      AuthorizerPayloadFormatVersion: "1.0"\n      EnableSimpleResponses: false\n      AuthorizerType: REQUEST\n      AuthorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n      IdentitySource:\n        - "hello.world.this.is.invalid"\n      Name: RequestAuthorizerV2Simple\n      ApiId: !Ref HttpLambdaAuth\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            return 10
        self.assertIn('Error: Lambda Authorizer RequestAuthorizerV2Simple does not contain valid identity sources.', self.start_api_process_output)