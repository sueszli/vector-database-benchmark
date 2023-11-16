import logging
from unittest.mock import patch
from twisted.test.proto_helpers import MemoryReactor
from synapse.rest import admin
from synapse.rest.client import login, room, sync
from synapse.server import HomeServer
from synapse.storage.util.id_generators import MultiWriterIdGenerator
from synapse.util import Clock
from tests.replication._base import BaseMultiWorkerStreamTestCase
from tests.server import make_request
logger = logging.getLogger(__name__)

class EventPersisterShardTestCase(BaseMultiWorkerStreamTestCase):
    """Checks event persisting sharding works"""
    servlets = [admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets, sync.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.other_user_id = self.register_user('otheruser', 'pass')
        self.other_access_token = self.login('otheruser', 'pass')
        self.room_creator = self.hs.get_room_creation_handler()
        self.store = hs.get_datastores().main

    def default_config(self) -> dict:
        if False:
            i = 10
            return i + 15
        conf = super().default_config()
        conf['stream_writers'] = {'events': ['worker1', 'worker2']}
        conf['instance_map'] = {'main': {'host': 'testserv', 'port': 8765}, 'worker1': {'host': 'testserv', 'port': 1001}, 'worker2': {'host': 'testserv', 'port': 1002}}
        return conf

    def _create_room(self, room_id: str, user_id: str, tok: str) -> None:
        if False:
            return 10
        'Create a room with given room_id'
        with patch('synapse.handlers.room.RoomCreationHandler._generate_room_id') as mock:
            mock.side_effect = lambda : room_id
            self.helper.create_room_as(user_id, tok=tok)

    def test_basic(self) -> None:
        if False:
            i = 10
            return i + 15
        'Simple test to ensure that multiple rooms can be created and joined,\n        and that different rooms get handled by different instances.\n        '
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker1'})
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker2'})
        persisted_on_1 = False
        persisted_on_2 = False
        store = self.hs.get_datastores().main
        user_id = self.register_user('user', 'pass')
        access_token = self.login('user', 'pass')
        for _ in range(10):
            room = self.helper.create_room_as(user_id, tok=access_token)
            self.helper.join(room=room, user=self.other_user_id, tok=self.other_access_token)
            rseponse = self.helper.send(room, body='Hi!', tok=self.other_access_token)
            event_id = rseponse['event_id']
            pos = self.get_success(store.get_position_for_event(event_id))
            persisted_on_1 |= pos.instance_name == 'worker1'
            persisted_on_2 |= pos.instance_name == 'worker2'
            if persisted_on_1 and persisted_on_2:
                break
        self.assertTrue(persisted_on_1)
        self.assertTrue(persisted_on_2)

    def test_vector_clock_token(self) -> None:
        if False:
            print('Hello World!')
        'Tests that using a stream token with a vector clock component works\n        correctly with basic /sync and /messages usage.\n        '
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker1'})
        worker_hs2 = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'worker2'})
        sync_hs = self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'sync'})
        sync_hs_site = self._hs_to_site[sync_hs]
        room_id1 = '!foo:test'
        room_id2 = '!baz:test'
        self.assertEqual(self.hs.config.worker.events_shard_config.get_instance(room_id1), 'worker1')
        self.assertEqual(self.hs.config.worker.events_shard_config.get_instance(room_id2), 'worker2')
        user_id = self.register_user('user', 'pass')
        access_token = self.login('user', 'pass')
        store = self.hs.get_datastores().main
        self._create_room(room_id1, user_id, access_token)
        self._create_room(room_id2, user_id, access_token)
        self.helper.join(room=room_id1, user=self.other_user_id, tok=self.other_access_token)
        self.helper.join(room=room_id2, user=self.other_user_id, tok=self.other_access_token)
        channel = make_request(self.reactor, sync_hs_site, 'GET', '/sync', access_token=access_token)
        next_batch = channel.json_body['next_batch']
        worker_store2 = worker_hs2.get_datastores().main
        assert isinstance(worker_store2._stream_id_gen, MultiWriterIdGenerator)
        actx = worker_store2._stream_id_gen.get_next()
        self.get_success(actx.__aenter__())
        response = self.helper.send(room_id1, body='Hi!', tok=self.other_access_token)
        first_event_in_room1 = response['event_id']
        room_stream_token = store.get_room_max_token()
        self.assertNotEqual(len(room_stream_token.instance_map), 0)
        channel = make_request(self.reactor, sync_hs_site, 'GET', f'/sync?since={next_batch}', access_token=access_token)
        self.assertIn(room_id1, channel.json_body['rooms']['join'])
        self.assertNotIn(room_id2, channel.json_body['rooms']['join'])
        events = channel.json_body['rooms']['join'][room_id1]['timeline']['events']
        self.assertListEqual([first_event_in_room1], [event['event_id'] for event in events])
        vector_clock_token = channel.json_body['next_batch']
        self.assertTrue(vector_clock_token.startswith('m'))
        self.get_success(actx.__aexit__(None, None, None))
        response = self.helper.send(room_id2, body='Hi!', tok=self.other_access_token)
        first_event_in_room2 = response['event_id']
        channel = make_request(self.reactor, sync_hs_site, 'GET', f'/sync?since={vector_clock_token}', access_token=access_token)
        self.assertNotIn(room_id1, channel.json_body['rooms']['join'])
        self.assertIn(room_id2, channel.json_body['rooms']['join'])
        events = channel.json_body['rooms']['join'][room_id2]['timeline']['events']
        self.assertListEqual([first_event_in_room2], [event['event_id'] for event in events])
        next_batch = channel.json_body['next_batch']
        self.helper.send(room_id1, body='Hi again!', tok=self.other_access_token)
        self.helper.send(room_id2, body='Hi again!', tok=self.other_access_token)
        channel = make_request(self.reactor, sync_hs_site, 'GET', f'/sync?since={next_batch}', access_token=access_token)
        prev_batch1 = channel.json_body['rooms']['join'][room_id1]['timeline']['prev_batch']
        prev_batch2 = channel.json_body['rooms']['join'][room_id2]['timeline']['prev_batch']
        channel = make_request(self.reactor, sync_hs_site, 'GET', '/rooms/{}/messages?from={}&to={}&dir=b'.format(room_id1, prev_batch1, vector_clock_token), access_token=access_token)
        self.assertListEqual([], channel.json_body['chunk'])
        channel = make_request(self.reactor, sync_hs_site, 'GET', '/rooms/{}/messages?from={}&to={}&dir=b'.format(room_id2, prev_batch2, vector_clock_token), access_token=access_token)
        self.assertEqual(len(channel.json_body['chunk']), 1)
        self.assertEqual(channel.json_body['chunk'][0]['event_id'], first_event_in_room2)
        channel = make_request(self.reactor, sync_hs_site, 'GET', '/rooms/{}/messages?from={}&to={}&dir=f'.format(room_id1, vector_clock_token, prev_batch1), access_token=access_token)
        self.assertListEqual([], channel.json_body['chunk'])
        channel = make_request(self.reactor, sync_hs_site, 'GET', '/rooms/{}/messages?from={}&to={}&dir=f'.format(room_id2, vector_clock_token, prev_batch2), access_token=access_token)
        self.assertEqual(len(channel.json_body['chunk']), 1)
        self.assertEqual(channel.json_body['chunk'][0]['event_id'], first_event_in_room2)