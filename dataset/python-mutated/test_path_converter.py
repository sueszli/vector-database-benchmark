from unittest import TestCase
from parameterized import parameterized
from samcli.local.apigw.path_converter import PathConverter

class TestPathConverter_toFlask(TestCase):

    def test_single_path_param(self):
        if False:
            for i in range(10):
                print('nop')
        path = '/{id}'
        flask_path = PathConverter.convert_path_to_flask(path)
        self.assertEqual(flask_path, '/<id>')

    def test_proxy_path(self):
        if False:
            print('Hello World!')
        path = '/{proxy+}'
        flask_path = PathConverter.convert_path_to_flask(path)
        self.assertEqual(flask_path, '/<path:proxy>')

    @parameterized.expand([('/{resource+}', '/<path:resource>'), ('/a/{id}/b/{resource+}', '/a/<id>/b/<path:resource>'), ('/a/b/{proxy}/{resource+}', '/a/b/<proxy>/<path:resource>'), ('/{id}/{something+}', '/<id>/<path:something>'), ('/{a}/{b}/{c}/{d+}', '/<a>/<b>/<c>/<path:d>')])
    def test_proxy_path_with_different_name(self, path, expected_result):
        if False:
            while True:
                i = 10
        flask_path = PathConverter.convert_path_to_flask(path)
        self.assertEqual(flask_path, expected_result)

    def test_proxy_with_path_param(self):
        if False:
            i = 10
            return i + 15
        path = '/id/{id}/user/{proxy+}'
        flask_path = PathConverter.convert_path_to_flask(path)
        self.assertEqual(flask_path, '/id/<id>/user/<path:proxy>')

    def test_multiple_path_params(self):
        if False:
            for i in range(10):
                print('nop')
        path = '/id/{id}/user/{user}'
        flask_path = PathConverter.convert_path_to_flask(path)
        self.assertEqual(flask_path, '/id/<id>/user/<user>')

    def test_no_changes_to_path(self):
        if False:
            i = 10
            return i + 15
        path = '/id/user'
        flask_path = PathConverter.convert_path_to_flask(path)
        self.assertEqual(flask_path, '/id/user')

class TestPathConverter_toApiGateway(TestCase):

    def test_single_path_param(self):
        if False:
            i = 10
            return i + 15
        path = '/<id>'
        flask_path = PathConverter.convert_path_to_api_gateway(path)
        self.assertEqual(flask_path, '/{id}')

    def test_proxy_path(self):
        if False:
            for i in range(10):
                print('nop')
        path = '/<path:proxy>'
        flask_path = PathConverter.convert_path_to_api_gateway(path)
        self.assertEqual(flask_path, '/{proxy+}')

    @parameterized.expand([('/<path:resource>', '/{resource+}'), ('/a/<id>/b/<path:resource>', '/a/{id}/b/{resource+}'), ('/a/b/<proxy>/<path:resource>', '/a/b/{proxy}/{resource+}'), ('/<id>/<path:something>', '/{id}/{something+}'), ('/<a>/<b>/<c>/<path:d>', '/{a}/{b}/{c}/{d+}')])
    def test_proxy_path_with_different_name(self, path, expected_result):
        if False:
            return 10
        flask_path = PathConverter.convert_path_to_api_gateway(path)
        self.assertEqual(flask_path, expected_result)

    def test_proxy_with_path_param(self):
        if False:
            i = 10
            return i + 15
        path = '/id/<id>/user/<path:proxy>'
        flask_path = PathConverter.convert_path_to_api_gateway(path)
        self.assertEqual(flask_path, '/id/{id}/user/{proxy+}')

    def test_multiple_path_params(self):
        if False:
            print('Hello World!')
        path = '/id/<id>/user/<user>'
        flask_path = PathConverter.convert_path_to_api_gateway(path)
        self.assertEqual(flask_path, '/id/{id}/user/{user}')

    def test_no_changes_to_path(self):
        if False:
            while True:
                i = 10
        path = '/id/user'
        flask_path = PathConverter.convert_path_to_api_gateway(path)
        self.assertEqual(flask_path, '/id/user')