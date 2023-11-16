import logging
from http import HTTPStatus
from parameterized import parameterized
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS
from synapse.config.server import DEFAULT_ROOM_VERSION
from synapse.events import EventBase, make_event_from_dict
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.storage.controllers.state import server_acl_evaluator_from_event
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config

class FederationServerTests(unittest.FederatingHomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    @parameterized.expand([(b'',), (b'foo',), (b'{"limit": Infinity}',)])
    def test_bad_request(self, query_content: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Querying with bad data returns a reasonable error code.\n        '
        u1 = self.register_user('u1', 'pass')
        u1_token = self.login('u1', 'pass')
        room_1 = self.helper.create_room_as(u1, tok=u1_token)
        self.inject_room_member(room_1, '@user:other.example.com', 'join')
        '/get_missing_events/(?P<room_id>[^/]*)/?'
        channel = self.make_request('POST', '/_matrix/federation/v1/get_missing_events/%s' % (room_1,), query_content)
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, channel.result)
        self.assertEqual(channel.json_body['errcode'], 'M_NOT_JSON')

class ServerACLsTestCase(unittest.TestCase):

    def test_blocked_server(self) -> None:
        if False:
            i = 10
            return i + 15
        e = _create_acl_event({'allow': ['*'], 'deny': ['evil.com']})
        logging.info('ACL event: %s', e.content)
        server_acl_evalutor = server_acl_evaluator_from_event(e)
        self.assertFalse(server_acl_evalutor.server_matches_acl_event('evil.com'))
        self.assertFalse(server_acl_evalutor.server_matches_acl_event('EVIL.COM'))
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('evil.com.au'))
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('honestly.not.evil.com'))

    def test_block_ip_literals(self) -> None:
        if False:
            while True:
                i = 10
        e = _create_acl_event({'allow_ip_literals': False, 'allow': ['*']})
        logging.info('ACL event: %s', e.content)
        server_acl_evalutor = server_acl_evaluator_from_event(e)
        self.assertFalse(server_acl_evalutor.server_matches_acl_event('1.2.3.4'))
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('1a.2.3.4'))
        self.assertFalse(server_acl_evalutor.server_matches_acl_event('[1:2::]'))
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('1:2:3:4'))

    def test_wildcard_matching(self) -> None:
        if False:
            i = 10
            return i + 15
        e = _create_acl_event({'allow': ['good*.com']})
        server_acl_evalutor = server_acl_evaluator_from_event(e)
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('good.com'), '* matches 0 characters')
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('GOOD.COM'), 'pattern is case-insensitive')
        self.assertTrue(server_acl_evalutor.server_matches_acl_event('good.aa.com'), "* matches several characters, including '.'")
        self.assertFalse(server_acl_evalutor.server_matches_acl_event('ishgood.com'), 'pattern does not allow prefixes')

class StateQueryTests(unittest.FederatingHomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def test_needs_to_be_in_room(self) -> None:
        if False:
            while True:
                i = 10
        '/v1/state/<room_id> requires the server to be in the room'
        u1 = self.register_user('u1', 'pass')
        u1_token = self.login('u1', 'pass')
        room_1 = self.helper.create_room_as(u1, tok=u1_token)
        channel = self.make_signed_federation_request('GET', '/_matrix/federation/v1/state/%s?event_id=xyz' % (room_1,))
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, channel.result)
        self.assertEqual(channel.json_body['errcode'], 'M_FORBIDDEN')

class SendJoinFederationTests(unittest.FederatingHomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        super().prepare(reactor, clock, hs)
        self._storage_controllers = hs.get_storage_controllers()
        creator_user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        self._room_id = self.helper.create_room_as(room_creator=creator_user_id, tok=tok)
        second_member_user_id = self.register_user('fozzie', 'bear')
        tok2 = self.login('fozzie', 'bear')
        self.helper.join(self._room_id, second_member_user_id, tok=tok2)

    def _make_join(self, user_id: str) -> JsonDict:
        if False:
            while True:
                i = 10
        channel = self.make_signed_federation_request('GET', f'/_matrix/federation/v1/make_join/{self._room_id}/{user_id}?ver={DEFAULT_ROOM_VERSION}')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        return channel.json_body

    def test_send_join(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'happy-path test of send_join'
        joining_user = '@misspiggy:' + self.OTHER_SERVER_NAME
        join_result = self._make_join(joining_user)
        join_event_dict = join_result['event']
        self.add_hashes_and_signatures_from_other_server(join_event_dict, KNOWN_ROOM_VERSIONS[DEFAULT_ROOM_VERSION])
        channel = self.make_signed_federation_request('PUT', f'/_matrix/federation/v2/send_join/{self._room_id}/x', content=join_event_dict)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        returned_state = [(ev['type'], ev['state_key']) for ev in channel.json_body['state']]
        self.assertCountEqual(returned_state, [('m.room.create', ''), ('m.room.power_levels', ''), ('m.room.join_rules', ''), ('m.room.history_visibility', ''), ('m.room.member', '@kermit:test'), ('m.room.member', '@fozzie:test')])
        returned_auth_chain_events = [(ev['type'], ev['state_key']) for ev in channel.json_body['auth_chain']]
        self.assertCountEqual(returned_auth_chain_events, [('m.room.create', ''), ('m.room.member', '@kermit:test'), ('m.room.power_levels', ''), ('m.room.join_rules', '')])
        r = self.get_success(self._storage_controllers.state.get_current_state(self._room_id))
        self.assertEqual(r['m.room.member', joining_user].membership, 'join')

    def test_send_join_partial_state(self) -> None:
        if False:
            return 10
        '/send_join should return partial state, if requested'
        joining_user = '@misspiggy:' + self.OTHER_SERVER_NAME
        join_result = self._make_join(joining_user)
        join_event_dict = join_result['event']
        self.add_hashes_and_signatures_from_other_server(join_event_dict, KNOWN_ROOM_VERSIONS[DEFAULT_ROOM_VERSION])
        channel = self.make_signed_federation_request('PUT', f'/_matrix/federation/v2/send_join/{self._room_id}/x?omit_members=true', content=join_event_dict)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        returned_state = [(ev['type'], ev['state_key']) for ev in channel.json_body['state']]
        self.assertCountEqual(returned_state, [('m.room.create', ''), ('m.room.power_levels', ''), ('m.room.join_rules', ''), ('m.room.history_visibility', ''), ('m.room.member', '@kermit:test'), ('m.room.member', '@fozzie:test')])
        returned_auth_chain_events = [(ev['type'], ev['state_key']) for ev in channel.json_body['auth_chain']]
        self.assertCountEqual(returned_auth_chain_events, [])
        r = self.get_success(self._storage_controllers.state.get_current_state(self._room_id))
        self.assertEqual(r['m.room.member', joining_user].membership, 'join')

    @override_config({'rc_joins_per_room': {'per_second': 0, 'burst_count': 3}})
    def test_make_join_respects_room_join_rate_limit(self) -> None:
        if False:
            print('Hello World!')
        joining_user = '@ronniecorbett:' + self.OTHER_SERVER_NAME
        self._make_join(joining_user)
        new_local_user = self.register_user('animal', 'animal')
        token = self.login('animal', 'animal')
        self.helper.join(self._room_id, new_local_user, tok=token)
        joining_user = '@ronniebarker:' + self.OTHER_SERVER_NAME
        channel = self.make_signed_federation_request('GET', f'/_matrix/federation/v1/make_join/{self._room_id}/{joining_user}?ver={DEFAULT_ROOM_VERSION}')
        self.assertEqual(channel.code, HTTPStatus.TOO_MANY_REQUESTS, channel.json_body)

    @override_config({'rc_joins_per_room': {'per_second': 0, 'burst_count': 3}})
    def test_send_join_contributes_to_room_join_rate_limit_and_is_limited(self) -> None:
        if False:
            while True:
                i = 10
        join_event_dicts = []
        for i in range(2):
            joining_user = f'@misspiggy{i}:{self.OTHER_SERVER_NAME}'
            join_result = self._make_join(joining_user)
            join_event_dict = join_result['event']
            self.add_hashes_and_signatures_from_other_server(join_event_dict, KNOWN_ROOM_VERSIONS[DEFAULT_ROOM_VERSION])
            join_event_dicts.append(join_event_dict)
        channel = self.make_signed_federation_request('PUT', f'/_matrix/federation/v2/send_join/{self._room_id}/join0', content=join_event_dicts[0])
        self.assertEqual(channel.code, 200, channel.json_body)
        channel = self.make_signed_federation_request('PUT', f'/_matrix/federation/v2/send_join/{self._room_id}/join1', content=join_event_dicts[1])
        self.assertEqual(channel.code, HTTPStatus.TOO_MANY_REQUESTS, channel.json_body)

def _create_acl_event(content: JsonDict) -> EventBase:
    if False:
        while True:
            i = 10
    return make_event_from_dict({'room_id': '!a:b', 'event_id': '$a:b', 'type': 'm.room.server_acls', 'sender': '@a:b', 'content': content})