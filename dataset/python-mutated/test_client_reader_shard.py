import logging
from synapse.rest.client import register
from tests.replication._base import BaseMultiWorkerStreamTestCase
from tests.server import make_request
logger = logging.getLogger(__name__)

class ClientReaderTestCase(BaseMultiWorkerStreamTestCase):
    """Test using one or more generic workers for registration."""
    servlets = [register.register_servlets]

    def _get_worker_hs_config(self) -> dict:
        if False:
            return 10
        config = self.default_config()
        config['worker_app'] = 'synapse.app.generic_worker'
        return config

    def test_register_single_worker(self) -> None:
        if False:
            while True:
                i = 10
        'Test that registration works when using a single generic worker.'
        worker_hs = self.make_worker_hs('synapse.app.generic_worker')
        site = self._hs_to_site[worker_hs]
        channel_1 = make_request(self.reactor, site, 'POST', 'register', {'username': 'user', 'type': 'm.login.password', 'password': 'bar'})
        self.assertEqual(channel_1.code, 401)
        session = channel_1.json_body['session']
        channel_2 = make_request(self.reactor, site, 'POST', 'register', {'auth': {'session': session, 'type': 'm.login.dummy'}})
        self.assertEqual(channel_2.code, 200)
        self.assertEqual(channel_2.json_body['user_id'], '@user:test')

    def test_register_multi_worker(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that registration works when using multiple generic workers.'
        worker_hs_1 = self.make_worker_hs('synapse.app.generic_worker')
        worker_hs_2 = self.make_worker_hs('synapse.app.generic_worker')
        site_1 = self._hs_to_site[worker_hs_1]
        channel_1 = make_request(self.reactor, site_1, 'POST', 'register', {'username': 'user', 'type': 'm.login.password', 'password': 'bar'})
        self.assertEqual(channel_1.code, 401)
        session = channel_1.json_body['session']
        site_2 = self._hs_to_site[worker_hs_2]
        channel_2 = make_request(self.reactor, site_2, 'POST', 'register', {'auth': {'session': session, 'type': 'm.login.dummy'}})
        self.assertEqual(channel_2.code, 200)
        self.assertEqual(channel_2.json_body['user_id'], '@user:test')