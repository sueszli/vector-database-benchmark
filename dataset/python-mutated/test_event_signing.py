from signedjson.key import decode_signing_key_base64
from signedjson.types import SigningKey
from synapse.api.room_versions import RoomVersions
from synapse.crypto.event_signing import add_hashes_and_signatures
from synapse.events import make_event_from_dict
from tests import unittest
SIGNING_KEY_SEED = 'YJDBA9Xnr2sVqXD9Vj7XVUnmFZcZrlw8Md7kMW+3XA1'
KEY_ALG = 'ed25519'
KEY_VER = '1'
KEY_NAME = '%s:%s' % (KEY_ALG, KEY_VER)
HOSTNAME = 'domain'

class EventSigningTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.signing_key: SigningKey = decode_signing_key_base64(KEY_ALG, KEY_VER, SIGNING_KEY_SEED)

    def test_sign_minimal(self) -> None:
        if False:
            i = 10
            return i + 15
        event_dict = {'event_id': '$0:domain', 'origin': 'domain', 'origin_server_ts': 1000000, 'signatures': {}, 'type': 'X', 'unsigned': {'age_ts': 1000000}}
        add_hashes_and_signatures(RoomVersions.V1, event_dict, HOSTNAME, self.signing_key)
        event = make_event_from_dict(event_dict)
        self.assertTrue(hasattr(event, 'hashes'))
        self.assertIn('sha256', event.hashes)
        self.assertEqual(event.hashes['sha256'], '6tJjLpXtggfke8UxFhAKg82QVkJzvKOVOOSjUDK4ZSI')
        self.assertTrue(hasattr(event, 'signatures'))
        self.assertIn(HOSTNAME, event.signatures)
        self.assertIn(KEY_NAME, event.signatures['domain'])
        self.assertEqual(event.signatures[HOSTNAME][KEY_NAME], '2Wptgo4CwmLo/Y8B8qinxApKaCkBG2fjTWB7AbP5Uy+aIbygsSdLOFzvdDjww8zUVKCmI02eP9xtyJxc/cLiBA')

    def test_sign_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        event_dict = {'content': {'body': 'Here is the message content'}, 'event_id': '$0:domain', 'origin': 'domain', 'origin_server_ts': 1000000, 'type': 'm.room.message', 'room_id': '!r:domain', 'sender': '@u:domain', 'signatures': {}, 'unsigned': {'age_ts': 1000000}}
        add_hashes_and_signatures(RoomVersions.V1, event_dict, HOSTNAME, self.signing_key)
        event = make_event_from_dict(event_dict)
        self.assertTrue(hasattr(event, 'hashes'))
        self.assertIn('sha256', event.hashes)
        self.assertEqual(event.hashes['sha256'], 'onLKD1bGljeBWQhWZ1kaP9SorVmRQNdN5aM2JYU2n/g')
        self.assertTrue(hasattr(event, 'signatures'))
        self.assertIn(HOSTNAME, event.signatures)
        self.assertIn(KEY_NAME, event.signatures['domain'])
        self.assertEqual(event.signatures[HOSTNAME][KEY_NAME], 'Wm+VzmOUOz08Ds+0NTWb1d4CZrVsJSikkeRxh6aCcUwu6pNC78FunoD7KNWzqFn241eYHYMGCA5McEiVPdhzBA')