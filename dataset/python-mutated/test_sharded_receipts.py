import logging
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import ReceiptTypes
from synapse.rest import admin
from synapse.rest.client import login, receipts, room, sync
from synapse.server import HomeServer
from synapse.storage.util.id_generators import MultiWriterIdGenerator
from synapse.types import StreamToken
from synapse.util import Clock
from tests.replication._base import BaseMultiWorkerStreamTestCase
from tests.server import make_request
logger = logging.getLogger(__name__)

class ReceiptsShardTestCase(BaseMultiWorkerStreamTestCase):
    """Checks receipts sharding works"""
    servlets = [admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets, sync.register_servlets, receipts.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.other_user_id = self.register_user('otheruser', 'pass')
        self.other_access_token = self.login('otheruser', 'pass')
        self.room_creator = self.hs.get_room_creation_handler()
        self.store = hs.get_datastores().main

    def default_config(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        conf = super().default_config()
        conf['stream_writers'] = {'receipts': ['worker1', 'worker2']}
        conf['instance_map'] = {'main': {'host': 'testserv', 'port': 8765}, 'worker1': {'host': 'testserv', 'port': 1001}, 'worker2': {'host': 'testserv', 'port': 1002}}
        return conf

    def test_basic(self) -> None:
        if False:
            return 10
        'Simple test to ensure that receipts can be sent on multiple\n        workers.\n        '
        worker1 = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker1'})
        worker1_site = self._hs_to_site[worker1]
        worker2 = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker2'})
        worker2_site = self._hs_to_site[worker2]
        user_id = self.register_user('user', 'pass')
        access_token = self.login('user', 'pass')
        room_id = self.helper.create_room_as(user_id, tok=access_token)
        self.helper.join(room=room_id, user=self.other_user_id, tok=self.other_access_token)
        response = self.helper.send(room_id, body='Hi!', tok=self.other_access_token)
        event_id = response['event_id']
        channel = make_request(reactor=self.reactor, site=worker1_site, method='POST', path=f'/rooms/{room_id}/receipt/{ReceiptTypes.READ}/{event_id}', access_token=access_token, content={})
        self.assertEqual(200, channel.code)
        response = self.helper.send(room_id, body='Hi!', tok=self.other_access_token)
        event_id = response['event_id']
        channel = make_request(reactor=self.reactor, site=worker2_site, method='POST', path=f'/rooms/{room_id}/receipt/{ReceiptTypes.READ}/{event_id}', access_token=access_token, content={})
        self.assertEqual(200, channel.code)

    def test_vector_clock_token(self) -> None:
        if False:
            print('Hello World!')
        'Tests that using a stream token with a vector clock component works\n        correctly with basic /sync usage.\n        '
        worker_hs1 = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker1'})
        worker1_site = self._hs_to_site[worker_hs1]
        worker_hs2 = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker2'})
        worker2_site = self._hs_to_site[worker_hs2]
        sync_hs = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'sync'})
        sync_hs_site = self._hs_to_site[sync_hs]
        user_id = self.register_user('user', 'pass')
        access_token = self.login('user', 'pass')
        store = self.hs.get_datastores().main
        room_id = self.helper.create_room_as(user_id, tok=access_token)
        self.helper.join(room=room_id, user=self.other_user_id, tok=self.other_access_token)
        response = self.helper.send(room_id, body='Hi!', tok=self.other_access_token)
        first_event = response['event_id']
        channel = make_request(self.reactor, sync_hs_site, 'GET', '/sync', access_token=access_token)
        next_batch = channel.json_body['next_batch']
        worker_store2 = worker_hs2.get_datastores().main
        assert isinstance(worker_store2._receipts_id_gen, MultiWriterIdGenerator)
        actx = worker_store2._receipts_id_gen.get_next()
        self.get_success(actx.__aenter__())
        channel = make_request(reactor=self.reactor, site=worker1_site, method='POST', path=f'/rooms/{room_id}/receipt/{ReceiptTypes.READ}/{first_event}', access_token=access_token, content={})
        self.assertEqual(200, channel.code)
        receipts_token = store.get_max_receipt_stream_id()
        self.assertGreater(len(receipts_token.instance_map), 0)
        channel = make_request(self.reactor, sync_hs_site, 'GET', f'/sync?since={next_batch}', access_token=access_token)
        self.assertIn(room_id, channel.json_body['rooms']['join'])
        events = channel.json_body['rooms']['join'][room_id]['ephemeral']['events']
        self.assertEqual(len(events), 1)
        self.assertIn(first_event, events[0]['content'])
        vector_clock_token = channel.json_body['next_batch']
        parsed_token = self.get_success(StreamToken.from_string(store, vector_clock_token))
        self.assertGreaterEqual(len(parsed_token.receipt_key.instance_map), 1)
        self.get_success(actx.__aexit__(None, None, None))
        response = self.helper.send(room_id, body='Hi!', tok=self.other_access_token)
        second_event = response['event_id']
        channel = make_request(reactor=self.reactor, site=worker2_site, method='POST', path=f'/rooms/{room_id}/receipt/{ReceiptTypes.READ}/{second_event}', access_token=access_token, content={})
        channel = make_request(self.reactor, sync_hs_site, 'GET', f'/sync?since={vector_clock_token}', access_token=access_token)
        self.assertIn(room_id, channel.json_body['rooms']['join'])
        events = channel.json_body['rooms']['join'][room_id]['ephemeral']['events']
        self.assertEqual(len(events), 1)
        self.assertIn(second_event, events[0]['content'])