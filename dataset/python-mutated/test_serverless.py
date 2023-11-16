import unittest
from troposphere import ImportValue, Parameter, Ref, Sub, Tags, Template
from troposphere.s3 import Filter, Rules, S3Key
from troposphere.serverless import SERVERLESS_TRANSFORM, Api, ApiEvent, ApiFunctionAuth, ApiGlobals, Auth, DeadLetterQueue, DeploymentPreference, Domain, EndpointConfiguration, Function, FunctionForPackaging, FunctionGlobals, FunctionUrlConfig, Globals, HttpApi, HttpApiAuth, HttpApiDomainConfiguration, HttpApiGlobals, LayerVersion, OAuth2Authorizer, ResourcePolicyStatement, Route53, S3Event, S3Location, SimpleTable, SimpleTableGlobals

class TestServerless(unittest.TestCase):

    def test_exactly_one_code(self):
        if False:
            return 10
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri=S3Location(Bucket='mybucket', Key='mykey'), InlineCode='')
        t = Template()
        t.add_resource(serverless_func)
        with self.assertRaises(ValueError):
            t.to_json()

    def test_s3_location(self):
        if False:
            return 10
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri=S3Location(Bucket='mybucket', Key='mykey'))
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()

    def test_tags(self):
        if False:
            while True:
                i = 10
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip', Tags=Tags({'Tag1': 'TagValue1', 'Tag2': 'TagValue2'}))
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()

    def test_DLQ(self):
        if False:
            while True:
                i = 10
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip', DeadLetterQueue=DeadLetterQueue(Type='SNS', TargetArn='arn:aws:sns:us-east-1:000000000000:SampleTopic'))
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()

    def test_required_function(self):
        if False:
            return 10
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip')
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()

    def test_functionurlconfig_oneof(self):
        if False:
            while True:
                i = 10
        serverless_func = Function('SomeHandler', FunctionUrlConfig=FunctionUrlConfig(AuthType='AWS_IAM'), Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip')
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()
        with self.assertRaises(ValueError):
            serverless_func = Function('SomeHandler', FunctionUrlConfig=FunctionUrlConfig(AuthType='foobar'), Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip')

    def test_optional_auto_publish_alias(self):
        if False:
            while True:
                i = 10
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip', AutoPublishAlias='alias')
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()

    def test_optional_deployment_preference(self):
        if False:
            for i in range(10):
                print('nop')
        serverless_func = Function('SomeHandler', Handler='index.handler', Runtime='nodejs', CodeUri='s3://bucket/handler.zip', AutoPublishAlias='alias', DeploymentPreference=DeploymentPreference(Type='AllAtOnce'))
        t = Template()
        t.add_resource(serverless_func)
        t.to_json()

    def test_required_api_definitionuri(self):
        if False:
            for i in range(10):
                print('nop')
        serverless_api = Api('SomeApi', StageName='test', DefinitionUri='s3://bucket/swagger.yml')
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()
    swagger = {'swagger': '2.0', 'info': {'title': 'swagger test'}, 'paths': {'/test': {'get': {}}}}

    def test_required_api_both(self):
        if False:
            while True:
                i = 10
        serverless_api = Api('SomeApi', StageName='test', DefinitionUri='s3://bucket/swagger.yml', DefinitionBody=self.swagger)
        t = Template()
        t.add_resource(serverless_api)
        with self.assertRaises(ValueError):
            t.to_json()

    def test_required_api_definitionbody(self):
        if False:
            return 10
        serverless_api = Api('SomeApi', StageName='test', DefinitionBody=self.swagger)
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()

    def test_api_no_definition(self):
        if False:
            return 10
        serverless_api = Api('SomeApi', StageName='test')
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()

    def test_api_auth_resource_policy(self):
        if False:
            for i in range(10):
                print('nop')
        serverless_api = Api(title='SomeApi', Auth=Auth(ResourcePolicy=ResourcePolicyStatement(AwsAccountBlacklist=['testAwsAccountBlacklist'], AwsAccountWhitelist=['testAwsAccountWhitelist'], CustomStatements=['testCustomStatements'], IpRangeBlacklist=['testIpRangeBlacklist'], IpRangeWhitelist=['testIpRangeWhitelist'], SourceVpcBlacklist=['testVpcBlacklist'], SourceVpcWhitelist=['testVpcWhitelist'])), StageName='testStageName')
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()

    def test_api_with_endpoint_configuration(self):
        if False:
            for i in range(10):
                print('nop')
        serverless_api = Api(title='SomeApi', StageName='testStageName', EndpointConfiguration=EndpointConfiguration(Type='PRIVATE'))
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()

    def test_api_with_domain(self):
        if False:
            return 10
        certificate = Parameter('certificate', Type='String')
        serverless_api = Api('SomeApi', StageName='test', Domain=Domain(BasePath=['/'], CertificateArn=Ref(certificate), DomainName=Sub('subdomain.${Zone}', Zone=ImportValue('MyZone')), EndpointConfiguration='REGIONAL', Route53=Route53(HostedZoneId=ImportValue('MyZone'), IpV6=True)))
        t = Template()
        t.add_parameter(certificate)
        t.add_resource(serverless_api)
        t.to_json()

    def test_http_api_definition_uri_defined(self):
        if False:
            print('Hello World!')
        serverless_http_api = HttpApi('SomeHttpApi', StageName='testHttp', DefinitionUri='s3://bucket/swagger.yml')
        t = Template()
        t.add_resource(serverless_http_api)
        t.to_json()

    def test_http_api_both_definition_uri_and_body_defined(self):
        if False:
            for i in range(10):
                print('nop')
        serverless_http_api = HttpApi('SomeHttpApi', StageName='testHttp', DefinitionUri='s3://bucket/swagger.yml', DefinitionBody=self.swagger)
        t = Template()
        t.add_resource(serverless_http_api)
        with self.assertRaises(ValueError):
            t.to_json()

    def test_http_api_definition_body(self):
        if False:
            print('Hello World!')
        serverless_http_api = HttpApi('SomeHttpApi', StageName='testHttp', DefinitionBody=self.swagger)
        t = Template()
        t.add_resource(serverless_http_api)
        t.to_json()

    def test_http_api_no_definition(self):
        if False:
            return 10
        serverless_api = HttpApi('SomeHttpApi', StageName='testHttp')
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()

    def test_http_api_authorization_scopes(self):
        if False:
            while True:
                i = 10
        serverless_api = HttpApi(title='SomeHttpApi', Auth=HttpApiAuth(Authorizers=OAuth2Authorizer(AuthorizationScopes=['scope1', 'scope2'])), StageName='testHttpStageName')
        t = Template()
        t.add_resource(serverless_api)
        t.to_json()

    def test_http_api_with_domain(self):
        if False:
            for i in range(10):
                print('nop')
        certificate = Parameter('certificate', Type='String')
        serverless_http_api = HttpApi('SomeHttpApi', StageName='testHttp', Domain=HttpApiDomainConfiguration(BasePath=['/'], CertificateArn=Ref(certificate), DomainName=Sub('subdomain.${Zone}', Zone=ImportValue('MyZone')), EndpointConfiguration='REGIONAL', Route53=Route53(HostedZoneId=ImportValue('MyZone'), IpV6=True)))
        t = Template()
        t.add_parameter(certificate)
        t.add_resource(serverless_http_api)
        t.to_json()

    def test_simple_table(self):
        if False:
            for i in range(10):
                print('nop')
        serverless_table = SimpleTable('SomeTable')
        t = Template()
        t.add_resource(serverless_table)
        t.to_json()

    def test_layer_version(self):
        if False:
            i = 10
            return i + 15
        layer_version = LayerVersion('SomeLayer', ContentUri='someuri')
        t = Template()
        t.add_resource(layer_version)
        t.to_json()
        layer_version = LayerVersion('SomeLayer')
        t = Template()
        t.add_resource(layer_version)
        with self.assertRaises(ValueError):
            t.to_json()

    def test_s3_filter(self):
        if False:
            for i in range(10):
                print('nop')
        t = Template()
        t.add_resource(Function('ProcessorFunction', Handler='process_file.handler', CodeUri='.', Runtime='python3.6', Policies='AmazonS3FullAccess', Events={'FileUpload': S3Event('FileUpload', Bucket='bucket', Events=['s3:ObjectCreated:*'], Filter=Filter(S3Key=S3Key(Rules=[Rules(Name='prefix', Value='upload/'), Rules(Name='suffix', Value='.txt')])))}))
        t.to_json()

    def test_policy_document(self):
        if False:
            for i in range(10):
                print('nop')
        t = Template()
        t.add_resource(Function('ProcessorFunction', Handler='process_file.handler', CodeUri='.', Runtime='python3.6', Policies='AmazonS3ReadOnly'))
        t.to_json()
        t = Template()
        t.add_resource(Function('ProcessorFunction', Handler='process_file.handler', CodeUri='.', Runtime='python3.6', Policies=['AmazonS3FullAccess', 'AmazonDynamoDBFullAccess']))
        t.to_json()
        t = Template()
        t.add_resource(Function('ProcessorFunction', Handler='process_file.handler', CodeUri='.', Runtime='python3.6', Policies={'Statement': [{'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:PutObject'], 'Resource': ['arn:aws:s3:::bucket/*']}]}))
        t.to_json()

    def test_packaging(self):
        if False:
            print('Hello World!')
        t = Template()
        t.add_resource(FunctionForPackaging('ProcessorFunction', Handler='process_file.handler', Runtime='python3.6', Policies={'Statement': [{'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:PutObject'], 'Resource': ['arn:aws:s3:::bucket/*']}]}))
        t.to_json()

    def test_globals(self):
        if False:
            print('Hello World!')
        t = Template()
        t.set_transform(SERVERLESS_TRANSFORM)
        t.set_globals(Globals(Function=FunctionGlobals(), Api=ApiGlobals(), HttpApi=HttpApiGlobals(), SimpleTable=SimpleTableGlobals()))
        t.to_json()
        with self.assertRaises(AttributeError):
            Globals(Unexpected='blah')
        with self.assertRaises(TypeError):
            Globals(Function='not FunctionGlobals')
        FunctionGlobals(Layers=['test'])
        with self.assertRaises(TypeError):
            FunctionGlobals(Layers='not a list')
        with self.assertRaises(TypeError):
            FunctionGlobals(Layers=[1, 2, 3])
        FunctionGlobals(MemorySize=128)
        with self.assertRaises(ValueError):
            FunctionGlobals(MemorySize=64)

    def test_api_event_auth(self):
        if False:
            print('Hello World!')
        api_event = ApiEvent('SomeApiEvent', Auth=Auth(), Path='some path', Method='some method')
        t = Template()
        t.add_resource(api_event)
        t.to_json()
        api_event = ApiEvent('SomeApiEvent', Auth=ApiFunctionAuth(), Path='some path', Method='some method')
        t = Template()
        t.add_resource(api_event)
        t.to_json()
        with self.assertRaises(TypeError):
            api_event = ApiEvent('SomeApiEvent', Auth='some auth', Path='some path', Method='some method')
if __name__ == '__main__':
    unittest.main()