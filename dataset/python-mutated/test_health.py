from synapse.rest.health import HealthResource
from tests import unittest

class HealthCheckTests(unittest.HomeserverTestCase):

    def create_test_resource(self) -> HealthResource:
        if False:
            return 10
        return HealthResource()

    def test_health(self) -> None:
        if False:
            print('Hello World!')
        channel = self.make_request('GET', '/health', shorthand=False)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.result['body'], b'OK')