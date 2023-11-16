from twisted.internet.defer import ensureDeferred
from synapse.rest.client import room
from tests.replication._base import BaseMultiWorkerStreamTestCase

class PartialStateStreamsTestCase(BaseMultiWorkerStreamTestCase):
    servlets = [room.register_servlets]
    hijack_auth = True
    user_id = '@bob:test'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.store = self.hs.get_datastores().main

    def test_un_partial_stated_room_unblocks_over_replication(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that, when a room is un-partial-stated on another worker,\n        pending calls to `await_full_state` get unblocked.\n        '
        room_id = self.helper.create_room_as('@bob:test')
        self.get_success(self.store.store_partial_state_room(room_id, {'serv1', 'serv2'}, 0, 'serv1'))
        worker = self.make_worker_hs('synapse.app.generic_worker')
        d = ensureDeferred(worker.get_storage_controllers().state.get_current_hosts_in_room(room_id))
        self.reactor.advance(0.1)
        self.assertFalse(d.called, 'get_current_hosts_in_room/await_full_state did not block')
        self.get_success(self.store.clear_partial_state_room(room_id))
        self.reactor.advance(0.1)
        self.assertTrue(d.called, 'get_current_hosts_in_room/await_full_state did not unblock')