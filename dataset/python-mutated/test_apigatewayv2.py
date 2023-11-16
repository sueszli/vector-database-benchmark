import unittest
from troposphere import Join
from troposphere.apigatewayv2 import Authorizer, IntegrationResponse, Model

class TestModel(unittest.TestCase):

    def test_schema(self):
        if False:
            while True:
                i = 10
        model = Model('schema', Name='model', ApiId='apiid')
        model.validate()
        model = Model('schema', ApiId='apiid', Name='model', Schema='{"a": "b"}')
        model.validate()
        model = Model('schema', ApiId='apiid', Name='model', Schema='{"a: "b"}')
        with self.assertRaises(ValueError):
            model.validate()
        d = {'c': 'd'}
        model = Model('schema', ApiId='apiid', Name='model', Schema=d)
        model.validate()
        self.assertEqual(model.properties['Schema'], '{"c": "d"}')
        with self.assertRaises(TypeError):
            model = Model('schema', ApiId='apiid', Name='model', Schema=1)
        model = Model('schema', ApiId='apiid', Name='model', Schema=Join(':', ['{"a', ': "b"}']))
        model.validate()

class TestIntegrationResponse(unittest.TestCase):

    def test_response_type(self):
        if False:
            print('Hello World!')
        integration_response = IntegrationResponse('IntegrationResponse', IntegrationId='integrationid', ApiId='apiid', IntegrationResponseKey='/400/')
        integration_response.validate()
        with self.assertRaises(ValueError):
            integration_response = IntegrationResponse('GatewayResponse', IntegrationId='integrationid', ApiId='apiid', IntegrationResponseKey='/400/', ContentHandlingStrategy='CONVERT_TO_SOMETHING')

class TestAuthorizer(unittest.TestCase):

    def test_response_type(self):
        if False:
            while True:
                i = 10
        authorizer = Authorizer('Authorizer', ApiId='apiid', AuthorizerType='REQUEST', AuthorizerUri='arn:lambda:function')
        authorizer.validate()
        with self.assertRaises(ValueError):
            authorizer = Authorizer('Authorizer', ApiId='apiid', AuthorizerType='RESPONSE', AuthorizerUri='arn:lambda:function')
if __name__ == '__main__':
    unittest.main()