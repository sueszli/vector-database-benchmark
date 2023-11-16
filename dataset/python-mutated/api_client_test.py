import time
import unittest
import warnings
import docker
from docker.utils import kwargs_from_env
from .base import BaseAPIIntegrationTest

class InformationTest(BaseAPIIntegrationTest):

    def test_version(self):
        if False:
            return 10
        res = self.client.version()
        assert 'GoVersion' in res
        assert 'Version' in res

    def test_info(self):
        if False:
            return 10
        res = self.client.info()
        assert 'Containers' in res
        assert 'Images' in res
        assert 'Debug' in res

class AutoDetectVersionTest(unittest.TestCase):

    def test_client_init(self):
        if False:
            i = 10
            return i + 15
        client = docker.APIClient(version='auto', **kwargs_from_env())
        client_version = client._version
        api_version = client.version(api_version=False)['ApiVersion']
        assert client_version == api_version
        api_version_2 = client.version()['ApiVersion']
        assert client_version == api_version_2
        client.close()

class ConnectionTimeoutTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.timeout = 0.5
        self.client = docker.api.APIClient(version=docker.constants.MINIMUM_DOCKER_API_VERSION, base_url='http://192.168.10.2:4243', timeout=self.timeout)

    def test_timeout(self):
        if False:
            while True:
                i = 10
        start = time.time()
        res = None
        try:
            res = self.client.inspect_container('id')
        except Exception:
            pass
        end = time.time()
        assert res is None
        assert end - start < 2 * self.timeout

class UnixconnTest(unittest.TestCase):
    """
    Test UNIX socket connection adapter.
    """

    def test_resource_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test no warnings are produced when using the client.\n        '
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            client = docker.APIClient(version='auto', **kwargs_from_env())
            client.images()
            client.close()
            del client
            assert len(w) == 0, f'No warnings produced: {w[0].message}'