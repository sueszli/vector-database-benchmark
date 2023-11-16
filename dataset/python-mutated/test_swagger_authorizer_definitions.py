import pytest
import requests
from tests.integration.local.start_api.start_api_integ_base import StartApiIntegBaseClass, WritableStartApiIntegBaseClass
from parameterized import parameterized_class

@parameterized_class(('template_path', 'endpoint', 'parameter_overrides'), [('/testdata/start_api/lambda_authorizers/swagger-api.yaml', '', {}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggertoken', {}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggertoken', {'ValidationString': '^myheader$'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggerrequest', {}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizeropenapi', {}), ('/testdata/start_api/lambda_authorizers/swagger-http.yaml', 'requestauthorizer', {}), ('/testdata/start_api/lambda_authorizers/swagger-http.yaml', 'requestauthorizersimple', {'AuthHandler': 'app.simple_handler'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggerrequest/authorized', {'AuthHandler': 'app.auth_handler_swagger_parameterized'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggertoken/authorized', {'AuthHandler': 'app.auth_handler_swagger_parameterized'})])
class TestSwaggerLambdaAuthorizerResources(StartApiIntegBaseClass):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_invokes_authorizer(self):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        response = requests.get(f'{self.url}/{self.endpoint}', timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response_json, {'message': 'Unauthorized'})

@parameterized_class(('template_path', 'endpoint', 'parameter_overrides'), [('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggertoken', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggerrequest', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizeropenapi', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/swagger-http.yaml', 'requestauthorizer', {'AuthHandler': 'app.unauth'}), ('/testdata/start_api/lambda_authorizers/swagger-http.yaml', 'requestauthorizersimple', {'AuthHandler': 'app.unauthv2'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggerrequest/unauthorized', {'AuthHandler': 'app.auth_handler_swagger_parameterized'}), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggertoken/unauthorized', {'AuthHandler': 'app.auth_handler_swagger_parameterized'})])
class TestSwaggerLambdaAuthorizersUnauthorized(StartApiIntegBaseClass):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.url = f'http://127.0.0.1:{self.port}'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=600, method='thread')
    def test_unauthorized_request(self):
        if False:
            for i in range(10):
                print('nop')
        headers = {'header': 'myheader'}
        query_string = {'query': 'myquery'}
        response = requests.get(f'{self.url}/{self.endpoint}', headers=headers, params=query_string, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response_json, {'message': 'User is not authorized to access this resource'})

@parameterized_class(('template_path', 'endpoint'), [('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggertoken'), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizerswaggerrequest'), ('/testdata/start_api/lambda_authorizers/swagger-api.yaml', 'requestauthorizeropenapi'), ('/testdata/start_api/lambda_authorizers/swagger-http.yaml', 'requestauthorizer'), ('/testdata/start_api/lambda_authorizers/swagger-http.yaml', 'requestauthorizersimple')])
class TestSwaggerLambdaAuthorizer500(StartApiIntegBaseClass):
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
            while True:
                i = 10
        headers = {'header': 'myheader'}
        query_string = {'query': 'myquery'}
        response = requests.get(f'{self.url}/{self.endpoint}', headers=headers, params=query_string, timeout=300)
        response_json = response.json()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response_json, {'message': 'Internal server error'})

class TestInvalidSwaggerTemplateUsingUnsupportedType(WritableStartApiIntegBaseClass):
    """
    Test using an invalid Lambda authorizer type
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  HttpApiOpenApi:\n    Type: AWS::ApiGatewayV2::Api\n    Properties:\n      Body:\n        openapi: "3.0"\n        info:\n          title: HttpApiOpenApi\n        components:\n          securitySchemes:\n            Authorizer:\n              type: apiKey\n              in: header\n              name: notused\n              "x-amazon-apigateway-authorizer":\n                authorizerPayloadFormatVersion: "2.0"\n                type: "bad type"\n                identitySource: "$request.header.header, $request.querystring.query"\n                authorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            i = 10
            return i + 15
        self.assertIn("Lambda authorizer 'Authorizer' type 'bad type' is unsupported, skipping", self.start_api_process_output)

class TestInvalidSwaggerTemplateUsingSimpleResponseWithPayloadV1(WritableStartApiIntegBaseClass):
    """
    Test using simple response with wrong payload version
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  HttpApiOpenApi:\n    Type: AWS::ApiGatewayV2::Api\n    Properties:\n      Body:\n        openapi: "3.0"\n        info:\n          title: HttpApiOpenApi\n        components:\n          securitySchemes:\n            Authorizer:\n              type: apiKey\n              in: header\n              name: notused\n              "x-amazon-apigateway-authorizer":\n                authorizerPayloadFormatVersion: "1.0"\n                type: "request"\n                enableSimpleResponses: True\n                identitySource: "$request.header.header, $request.querystring.query"\n                authorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIn("Simple responses are only available on HTTP APIs with payload version 2.0, ignoring for Lambda authorizer 'Authorizer'", self.start_api_process_output)

class TestInvalidSwaggerTemplateUsingUnsupportedPayloadVersion(WritableStartApiIntegBaseClass):
    """
    Test using an incorrect payload format version
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  HttpApiOpenApi:\n    Type: AWS::ApiGatewayV2::Api\n    Properties:\n      Body:\n        openapi: "3.0"\n        info:\n          title: HttpApiOpenApi\n        components:\n          securitySchemes:\n            Authorizer:\n              type: apiKey\n              in: header\n              name: notused\n              "x-amazon-apigateway-authorizer":\n                authorizerPayloadFormatVersion: "1.2.3"\n                type: "request"\n                identitySource: "$request.header.header, $request.querystring.query"\n                authorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            return 10
        self.assertIn("Error: Authorizer 'Authorizer' contains an invalid payload version", self.start_api_process_output)

class TestInvalidSwaggerTemplateUsingInvalidIdentitySources(WritableStartApiIntegBaseClass):
    """
    Test using an invalid identity source (a.b.c.d.e)
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  HttpApiOpenApi:\n    Type: AWS::ApiGatewayV2::Api\n    Properties:\n      Body:\n        openapi: "3.0"\n        info:\n          title: HttpApiOpenApi\n        components:\n          securitySchemes:\n            Authorizer:\n              type: apiKey\n              in: header\n              name: notused\n              "x-amazon-apigateway-authorizer":\n                authorizerPayloadFormatVersion: "2.0"\n                type: "request"\n                identitySource: "a.b.c.d.e"\n                authorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            return 10
        self.assertIn("Error: Identity source 'a.b.c.d.e' for Lambda Authorizer 'Authorizer' is not a valid identity source, check the spelling/format.", self.start_api_process_output)

class TestInvalidSwaggerTemplateUsingTokenWithHttpApi(WritableStartApiIntegBaseClass):
    """
    Test using token authorizer with HTTP API
    """
    do_collect_cmd_init_output = True
    template_content = 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\n\nResources:\n  HttpApiOpenApi:\n    Type: AWS::ApiGatewayV2::Api\n    Properties:\n      Body:\n        openapi: "3.0"\n        info:\n          title: HttpApiOpenApi\n        components:\n          securitySchemes:\n            Authorizer:\n              type: apiKey\n              in: header\n              name: notused\n              "x-amazon-apigateway-authorizer":\n                authorizerPayloadFormatVersion: "2.0"\n                type: "token"\n                identitySource: "$request.header.header"\n                authorizerUri: arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123456789012:function:AuthorizerFunction/invocations\n'

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.timeout(timeout=10, method='thread')
    def test_invalid_template(self):
        if False:
            i = 10
            return i + 15
        self.assertIn("Type 'token' for Lambda Authorizer 'Authorizer' is unsupported", self.start_api_process_output)