from typing import Callable, FrozenSet, List, Optional, Set
from unittest.mock import AsyncMock, Mock
from signedjson import key, sign
from signedjson.types import BaseKey, SigningKey
from twisted.internet import defer
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EduTypes, RoomEncryptionAlgorithms
from synapse.federation.units import Transaction
from synapse.handlers.device import DeviceHandler
from synapse.rest import admin
from synapse.rest.client import login
from synapse.server import HomeServer
from synapse.types import JsonDict, ReadReceipt
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class FederationSenderReceiptsTestCases(HomeserverTestCase):
    """
    Test federation sending to update receipts.

    By default for test cases federation sending is disabled. This Test class has it
    re-enabled for the main process.
    """

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        self.federation_transport_client = Mock(spec=['send_transaction'])
        self.federation_transport_client.send_transaction = AsyncMock()
        hs = self.setup_test_homeserver(federation_transport_client=self.federation_transport_client)
        hs.get_storage_controllers().state.get_current_hosts_in_room = AsyncMock(return_value={'test', 'host2'})
        hs.get_storage_controllers().state.get_current_hosts_in_room_or_partial_state_approximation = hs.get_storage_controllers().state.get_current_hosts_in_room
        return hs

    def default_config(self) -> JsonDict:
        if False:
            for i in range(10):
                print('nop')
        config = super().default_config()
        config['federation_sender_instances'] = None
        return config

    def test_send_receipts(self) -> None:
        if False:
            while True:
                i = 10
        mock_send_transaction = self.federation_transport_client.send_transaction
        mock_send_transaction.return_value = {}
        sender = self.hs.get_federation_sender()
        receipt = ReadReceipt('room_id', 'm.read', 'user_id', ['event_id'], thread_id=None, data={'ts': 1234})
        self.get_success(sender.send_read_receipt(receipt))
        self.pump()
        mock_send_transaction.assert_called_once()
        json_cb = mock_send_transaction.call_args[0][1]
        data = json_cb()
        self.assertEqual(data['edus'], [{'edu_type': EduTypes.RECEIPT, 'content': {'room_id': {'m.read': {'user_id': {'event_ids': ['event_id'], 'data': {'ts': 1234}}}}}}])

    def test_send_receipts_thread(self) -> None:
        if False:
            i = 10
            return i + 15
        mock_send_transaction = self.federation_transport_client.send_transaction
        mock_send_transaction.return_value = {}
        sender = self.hs.get_federation_sender()
        sender.wake_destination('host2')
        for (user, thread) in (('alice', None), ('alice', 'thread'), ('bob', None), ('bob', 'diff-thread')):
            receipt = ReadReceipt('room_id', 'm.read', user, ['event_id'], thread_id=thread, data={'ts': 1234})
            defer.ensureDeferred(sender.send_read_receipt(receipt))
        self.pump()
        mock_send_transaction.assert_called_once()
        json_cb = mock_send_transaction.call_args[0][1]
        data = json_cb()
        self.assertCountEqual(data['edus'], [{'edu_type': EduTypes.RECEIPT, 'content': {'room_id': {'m.read': {'alice': {'event_ids': ['event_id'], 'data': {'ts': 1234, 'thread_id': 'thread'}}, 'bob': {'event_ids': ['event_id'], 'data': {'ts': 1234, 'thread_id': 'diff-thread'}}}}}}, {'edu_type': EduTypes.RECEIPT, 'content': {'room_id': {'m.read': {'alice': {'event_ids': ['event_id'], 'data': {'ts': 1234}}, 'bob': {'event_ids': ['event_id'], 'data': {'ts': 1234}}}}}}])

    def test_send_receipts_with_backoff(self) -> None:
        if False:
            while True:
                i = 10
        'Send two receipts in quick succession; the second should be flushed, but\n        only after 20ms'
        mock_send_transaction = self.federation_transport_client.send_transaction
        mock_send_transaction.return_value = {}
        sender = self.hs.get_federation_sender()
        receipt = ReadReceipt('room_id', 'm.read', 'user_id', ['event_id'], thread_id=None, data={'ts': 1234})
        self.get_success(sender.send_read_receipt(receipt))
        self.pump()
        mock_send_transaction.assert_called_once()
        json_cb = mock_send_transaction.call_args[0][1]
        data = json_cb()
        self.assertEqual(data['edus'], [{'edu_type': EduTypes.RECEIPT, 'content': {'room_id': {'m.read': {'user_id': {'event_ids': ['event_id'], 'data': {'ts': 1234}}}}}}])
        mock_send_transaction.reset_mock()
        receipt = ReadReceipt('room_id', 'm.read', 'user_id', ['other_id'], thread_id=None, data={'ts': 1234})
        self.successResultOf(defer.ensureDeferred(sender.send_read_receipt(receipt)))
        self.pump()
        mock_send_transaction.assert_not_called()
        self.reactor.advance(19)
        mock_send_transaction.assert_not_called()
        self.reactor.advance(10)
        mock_send_transaction.assert_called_once()
        json_cb = mock_send_transaction.call_args[0][1]
        data = json_cb()
        self.assertEqual(data['edus'], [{'edu_type': EduTypes.RECEIPT, 'content': {'room_id': {'m.read': {'user_id': {'event_ids': ['other_id'], 'data': {'ts': 1234}}}}}}])

class FederationSenderDevicesTestCases(HomeserverTestCase):
    """
    Test federation sending to update devices.

    By default for test cases federation sending is disabled. This Test class has it
    re-enabled for the main process.
    """
    servlets = [admin.register_servlets, login.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        self.federation_transport_client = Mock(spec=['send_transaction', 'query_user_devices'])
        self.federation_transport_client.send_transaction = AsyncMock()
        self.federation_transport_client.query_user_devices = AsyncMock()
        return self.setup_test_homeserver(federation_transport_client=self.federation_transport_client)

    def default_config(self) -> JsonDict:
        if False:
            return 10
        c = super().default_config()
        c['federation_sender_instances'] = None
        return c

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        test_room_id = '!room:host1'

        def get_rooms_for_user(user_id: str) -> 'defer.Deferred[FrozenSet[str]]':
            if False:
                return 10
            return defer.succeed(frozenset({test_room_id}))
        hs.get_datastores().main.get_rooms_for_user = get_rooms_for_user

        async def get_current_hosts_in_room(room_id: str) -> Set[str]:
            if room_id == test_room_id:
                return {'host2'}
            else:
                return set()
        hs.get_datastores().main.get_current_hosts_in_room = get_current_hosts_in_room
        device_handler = hs.get_device_handler()
        assert isinstance(device_handler, DeviceHandler)
        self.device_handler = device_handler
        self.edus: List[JsonDict] = []
        self.federation_transport_client.send_transaction.side_effect = self.record_transaction

    async def record_transaction(self, txn: Transaction, json_cb: Optional[Callable[[], JsonDict]]=None) -> JsonDict:
        assert json_cb is not None
        data = json_cb()
        self.edus.extend(data['edus'])
        return {}

    def test_send_device_updates(self) -> None:
        if False:
            while True:
                i = 10
        'Basic case: each device update should result in an EDU'
        u1 = self.register_user('user', 'pass')
        self.login(u1, 'pass', device_id='D1')
        self.assertEqual(len(self.edus), 1)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D1', None)
        self.reactor.advance(1)
        self.get_success(self.hs.get_federation_sender().send_device_messages(['host2']))
        self.assertEqual(self.edus, [])
        self.login('user', 'pass', device_id='D2')
        self.assertEqual(len(self.edus), 1)
        self.check_device_update_edu(self.edus.pop(0), u1, 'D2', stream_id)

    def test_dont_send_device_updates_for_remote_users(self) -> None:
        if False:
            return 10
        "Check that we don't send device updates for remote users"
        self.federation_transport_client.query_user_devices.return_value = {'stream_id': '1', 'user_id': '@user2:host2', 'devices': [{'device_id': 'D1'}]}
        self.get_success(self.device_handler.device_list_updater.incoming_device_list_update('host2', {'user_id': '@user2:host2', 'device_id': 'D1', 'stream_id': '1', 'prev_ids': []}))
        self.reactor.advance(1)
        self.assertEqual(self.edus, [])
        devices = self.get_success(self.hs.get_datastores().main.get_cached_devices_for_user('@user2:host2'))
        self.assertIn('D1', devices)

    def test_upload_signatures(self) -> None:
        if False:
            print('Hello World!')
        'Uploading signatures on some devices should produce updates for that user'
        e2e_handler = self.hs.get_e2e_keys_handler()
        u1 = self.register_user('user', 'pass')
        self.login(u1, 'pass', device_id='D1')
        self.login(u1, 'pass', device_id='D2')
        self.assertEqual(len(self.edus), 2)
        stream_id: Optional[int] = None
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D1', stream_id)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D2', stream_id)
        device1_signing_key = self.generate_and_upload_device_signing_key(u1, 'D1')
        device2_signing_key = self.generate_and_upload_device_signing_key(u1, 'D2')
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 2)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D1', stream_id)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D2', stream_id)
        master_signing_key = generate_self_id_key()
        master_key = {'user_id': u1, 'usage': ['master'], 'keys': {key_id(master_signing_key): encode_pubkey(master_signing_key)}}
        selfsigning_signing_key = generate_self_id_key()
        selfsigning_key = {'user_id': u1, 'usage': ['self_signing'], 'keys': {key_id(selfsigning_signing_key): encode_pubkey(selfsigning_signing_key)}}
        sign.sign_json(selfsigning_key, u1, master_signing_key)
        cross_signing_keys = {'master_key': master_key, 'self_signing_key': selfsigning_key}
        self.get_success(e2e_handler.upload_signing_keys_for_user(u1, cross_signing_keys))
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 2)
        self.assertEqual(self.edus.pop(0)['edu_type'], EduTypes.SIGNING_KEY_UPDATE)
        self.assertEqual(self.edus.pop(0)['edu_type'], EduTypes.UNSTABLE_SIGNING_KEY_UPDATE)
        d1_json = build_device_dict(u1, 'D1', device1_signing_key)
        sign.sign_json(d1_json, u1, selfsigning_signing_key)
        d2_json = build_device_dict(u1, 'D2', device2_signing_key)
        sign.sign_json(d2_json, u1, selfsigning_signing_key)
        ret = self.get_success(e2e_handler.upload_signatures_for_device_keys(u1, {u1: {'D1': d1_json, 'D2': d2_json}}))
        self.assertEqual(ret['failures'], {})
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 2)
        stream_id = None
        for edu in self.edus:
            self.assertEqual(edu['edu_type'], EduTypes.DEVICE_LIST_UPDATE)
            c = edu['content']
            if stream_id is not None:
                self.assertEqual(c['prev_id'], [stream_id])
                self.assertGreaterEqual(c['stream_id'], stream_id)
            stream_id = c['stream_id']
        devices = {edu['content']['device_id'] for edu in self.edus}
        self.assertEqual({'D1', 'D2'}, devices)

    def test_delete_devices(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'If devices are deleted, that should result in EDUs too'
        u1 = self.register_user('user', 'pass')
        self.login('user', 'pass', device_id='D1')
        self.login('user', 'pass', device_id='D2')
        self.login('user', 'pass', device_id='D3')
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 3)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D1', None)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D2', stream_id)
        stream_id = self.check_device_update_edu(self.edus.pop(0), u1, 'D3', stream_id)
        self.get_success(self.device_handler.delete_devices(u1, ['D1', 'D2', 'D3']))
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 3)
        for edu in self.edus:
            self.assertEqual(edu['edu_type'], EduTypes.DEVICE_LIST_UPDATE)
            c = edu['content']
            self.assertGreaterEqual(c.items(), {'user_id': u1, 'prev_id': [stream_id], 'deleted': True}.items())
            self.assertGreaterEqual(c['stream_id'], stream_id)
            stream_id = c['stream_id']
        devices = {edu['content']['device_id'] for edu in self.edus}
        self.assertEqual({'D1', 'D2', 'D3'}, devices)

    def test_unreachable_server(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'If the destination server is unreachable, all the updates should get sent on\n        recovery\n        '
        mock_send_txn = self.federation_transport_client.send_transaction
        mock_send_txn.side_effect = AssertionError('fail')
        u1 = self.register_user('user', 'pass')
        self.login('user', 'pass', device_id='D1')
        self.login('user', 'pass', device_id='D2')
        self.login('user', 'pass', device_id='D3')
        self.get_success(self.device_handler.delete_devices(u1, ['D1', 'D2', 'D3']))
        self.reactor.advance(1)
        self.assertGreaterEqual(mock_send_txn.call_count, 4)
        mock_send_txn.side_effect = self.record_transaction
        self.get_success(self.hs.get_federation_sender().send_device_messages(['host2']))
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 3)
        stream_id: Optional[int] = None
        for edu in self.edus:
            self.assertEqual(edu['edu_type'], EduTypes.DEVICE_LIST_UPDATE)
            c = edu['content']
            self.assertEqual(c['prev_id'], [stream_id] if stream_id is not None else [])
            if stream_id is not None:
                self.assertGreaterEqual(c['stream_id'], stream_id)
            stream_id = c['stream_id']
        devices = {edu['content']['device_id'] for edu in self.edus}
        self.assertEqual({'D1', 'D2', 'D3'}, devices)

    def test_prune_outbound_device_pokes1(self) -> None:
        if False:
            i = 10
            return i + 15
        'If a destination is unreachable, and the updates are pruned, we should get\n        a single update.\n\n        This case tests the behaviour when the server has never been reachable.\n        '
        mock_send_txn = self.federation_transport_client.send_transaction
        mock_send_txn.side_effect = AssertionError('fail')
        u1 = self.register_user('user', 'pass')
        self.login('user', 'pass', device_id='D1')
        self.login('user', 'pass', device_id='D2')
        self.login('user', 'pass', device_id='D3')
        self.get_success(self.device_handler.delete_devices(u1, ['D1', 'D2', 'D3']))
        self.reactor.advance(1)
        self.assertGreaterEqual(mock_send_txn.call_count, 4)
        self.reactor.advance(10)
        self.get_success(self.hs.get_datastores().main._prune_old_outbound_device_pokes(prune_age=1))
        mock_send_txn.side_effect = self.record_transaction
        self.get_success(self.hs.get_federation_sender().send_device_messages(['host2']))
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 1)
        edu = self.edus.pop(0)
        self.assertEqual(edu['edu_type'], EduTypes.DEVICE_LIST_UPDATE)
        c = edu['content']
        self.assertEqual(c['prev_id'], [])

    def test_prune_outbound_device_pokes2(self) -> None:
        if False:
            i = 10
            return i + 15
        'If a destination is unreachable, and the updates are pruned, we should get\n        a single update.\n\n        This case tests the behaviour when the server was reachable, but then goes\n        offline.\n        '
        u1 = self.register_user('user', 'pass')
        self.login('user', 'pass', device_id='D1')
        self.assertEqual(len(self.edus), 1)
        self.check_device_update_edu(self.edus.pop(0), u1, 'D1', None)
        mock_send_txn = self.federation_transport_client.send_transaction
        mock_send_txn.side_effect = AssertionError('fail')
        self.login('user', 'pass', device_id='D2')
        self.login('user', 'pass', device_id='D3')
        self.reactor.advance(1)
        self.get_success(self.device_handler.delete_devices(u1, ['D1', 'D2', 'D3']))
        self.assertGreaterEqual(mock_send_txn.call_count, 3)
        self.reactor.advance(10)
        self.get_success(self.hs.get_datastores().main._prune_old_outbound_device_pokes(prune_age=1))
        mock_send_txn.side_effect = self.record_transaction
        self.get_success(self.hs.get_federation_sender().send_device_messages(['host2']))
        self.reactor.advance(1)
        self.assertEqual(len(self.edus), 1)
        edu = self.edus.pop(0)
        self.assertEqual(edu['edu_type'], EduTypes.DEVICE_LIST_UPDATE)
        c = edu['content']
        self.assertEqual(c['prev_id'], [])

    def check_device_update_edu(self, edu: JsonDict, user_id: str, device_id: str, prev_stream_id: Optional[int]) -> int:
        if False:
            print('Hello World!')
        'Check that the given EDU is an update for the given device\n        Returns the stream_id.\n        '
        self.assertEqual(edu['edu_type'], EduTypes.DEVICE_LIST_UPDATE)
        content = edu['content']
        expected = {'user_id': user_id, 'device_id': device_id, 'prev_id': [prev_stream_id] if prev_stream_id is not None else []}
        self.assertLessEqual(expected.items(), content.items())
        if prev_stream_id is not None:
            self.assertGreaterEqual(content['stream_id'], prev_stream_id)
        return content['stream_id']

    def check_signing_key_update_txn(self, txn: JsonDict) -> None:
        if False:
            while True:
                i = 10
        'Check that the txn has an EDU with a signing key update.'
        edus = txn['edus']
        self.assertEqual(len(edus), 2)

    def generate_and_upload_device_signing_key(self, user_id: str, device_id: str) -> SigningKey:
        if False:
            i = 10
            return i + 15
        'Generate a signing keypair for the given device, and upload it'
        sk = key.generate_signing_key(device_id)
        device_dict = build_device_dict(user_id, device_id, sk)
        self.get_success(self.hs.get_e2e_keys_handler().upload_keys_for_user(user_id, device_id, {'device_keys': device_dict}))
        return sk

def generate_self_id_key() -> SigningKey:
    if False:
        return 10
    'generate a signing key whose version is its public key\n\n    ... as used by the cross-signing-keys.\n    '
    k = key.generate_signing_key('x')
    k.version = encode_pubkey(k)
    return k

def key_id(k: BaseKey) -> str:
    if False:
        print('Hello World!')
    return '%s:%s' % (k.alg, k.version)

def encode_pubkey(sk: SigningKey) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Encode the public key corresponding to the given signing key as base64'
    return key.encode_verify_key_base64(key.get_verify_key(sk))

def build_device_dict(user_id: str, device_id: str, sk: SigningKey) -> JsonDict:
    if False:
        print('Hello World!')
    'Build a dict representing the given device'
    return {'user_id': user_id, 'device_id': device_id, 'algorithms': ['m.olm.curve25519-aes-sha2', RoomEncryptionAlgorithms.MEGOLM_V1_AES_SHA2], 'keys': {'curve25519:' + device_id: 'curve25519+key', key_id(sk): encode_pubkey(sk)}}