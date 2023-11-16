import logging
import synapse
from synapse.replication.tcp.streams._base import _STREAM_UPDATE_TARGET_ROW_COUNT
from synapse.types import JsonDict
from tests.replication._base import BaseStreamTestCase
logger = logging.getLogger(__name__)

class ToDeviceStreamTestCase(BaseStreamTestCase):
    servlets = [synapse.rest.admin.register_servlets, synapse.rest.client.login.register_servlets]

    def test_to_device_stream(self) -> None:
        if False:
            i = 10
            return i + 15
        store = self.hs.get_datastores().main
        user1 = self.register_user('user1', 'pass')
        self.login('user1', 'pass', 'device')
        user2 = self.register_user('user2', 'pass')
        self.login('user2', 'pass', 'device')
        self.reconnect()
        self.replicate()
        self.test_handler.received_rdata_rows.clear()
        self.disconnect()
        msg: JsonDict = {}
        msg['sender'] = '@sender:example.org'
        msg['type'] = 'm.new_device'
        for i in range(_STREAM_UPDATE_TARGET_ROW_COUNT):
            msg['content'] = {'device': {}}
            messages = {user1: {'device': msg}}
            self.get_success(store.add_messages_from_remote_to_device_inbox('example.org', f'{i}', messages))
        msg['content'] = {'device': {}}
        messages = {user2: {'device': msg}}
        self.get_success(store.add_messages_from_remote_to_device_inbox('example.org', f'{_STREAM_UPDATE_TARGET_ROW_COUNT}', messages))
        self.assertEqual([], self.test_handler.received_rdata_rows)
        self.reconnect()
        self.replicate()
        received_rows = self.test_handler.received_rdata_rows
        self.assertEqual(len(received_rows), 2)
        self.assertEqual(received_rows[0][2].entity, user1)
        self.assertEqual(received_rows[1][2].entity, user2)