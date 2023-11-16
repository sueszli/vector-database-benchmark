from typing import List, Optional
from parameterized import parameterized
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventTypes, RelationTypes
from synapse.api.room_versions import RoomVersion, RoomVersions
from synapse.rest import admin
from synapse.rest.client import login, room, sync
from synapse.server import HomeServer
from synapse.storage._base import db_to_json
from synapse.storage.database import LoggingTransaction
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase, override_config

class RedactionsTestCase(HomeserverTestCase):
    """Tests that various redaction events are handled correctly"""
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets, sync.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        config = self.default_config()
        config['rc_message'] = {'per_second': 0.2, 'burst_count': 10}
        config['rc_admin_redaction'] = {'per_second': 1, 'burst_count': 100}
        return self.setup_test_homeserver(config=config)

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.mod_user_id = self.register_user('user1', 'pass')
        self.mod_access_token = self.login('user1', 'pass')
        self.other_user_id = self.register_user('otheruser', 'pass')
        self.other_access_token = self.login('otheruser', 'pass')
        self.room_id = self.helper.create_room_as(self.mod_user_id, tok=self.mod_access_token)
        self.helper.invite(room=self.room_id, src=self.mod_user_id, tok=self.mod_access_token, targ=self.other_user_id)
        self.helper.join(room=self.room_id, user=self.other_user_id, tok=self.other_access_token)

    def _redact_event(self, access_token: str, room_id: str, event_id: str, expect_code: int=200, with_relations: Optional[List[str]]=None, content: Optional[JsonDict]=None) -> JsonDict:
        if False:
            print('Hello World!')
        'Helper function to send a redaction event.\n\n        Returns the json body.\n        '
        path = '/_matrix/client/r0/rooms/%s/redact/%s' % (room_id, event_id)
        request_content = content or {}
        if with_relations:
            request_content['org.matrix.msc3912.with_relations'] = with_relations
        channel = self.make_request('POST', path, request_content, access_token=access_token)
        self.assertEqual(channel.code, expect_code)
        return channel.json_body

    def _sync_room_timeline(self, access_token: str, room_id: str) -> List[JsonDict]:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('GET', 'sync', access_token=access_token)
        self.assertEqual(channel.code, 200)
        room_sync = channel.json_body['rooms']['join'][room_id]
        return room_sync['timeline']['events']

    def test_redact_event_as_moderator(self) -> None:
        if False:
            return 10
        b = self.helper.send(room_id=self.room_id, tok=self.other_access_token)
        msg_id = b['event_id']
        b = self._redact_event(self.mod_access_token, self.room_id, msg_id)
        redaction_id = b['event_id']
        timeline = self._sync_room_timeline(self.mod_access_token, self.room_id)
        self.assertEqual(timeline[-1]['event_id'], redaction_id)
        self.assertEqual(timeline[-1]['redacts'], msg_id)
        self.assertEqual(timeline[-2]['event_id'], msg_id)
        self.assertEqual(timeline[-2]['unsigned']['redacted_by'], redaction_id)
        self.assertEqual(timeline[-2]['content'], {})

    def test_redact_event_as_normal(self) -> None:
        if False:
            return 10
        b = self.helper.send(room_id=self.room_id, tok=self.other_access_token)
        normal_msg_id = b['event_id']
        b = self.helper.send(room_id=self.room_id, tok=self.mod_access_token)
        admin_msg_id = b['event_id']
        self._redact_event(self.other_access_token, self.room_id, admin_msg_id, expect_code=403)
        b = self._redact_event(self.other_access_token, self.room_id, normal_msg_id)
        redaction_id = b['event_id']
        timeline = self._sync_room_timeline(self.other_access_token, self.room_id)
        self.assertEqual(timeline[-1]['event_id'], redaction_id)
        self.assertEqual(timeline[-1]['redacts'], normal_msg_id)
        self.assertEqual(timeline[-2]['event_id'], admin_msg_id)
        self.assertNotIn('redacted_by', timeline[-2]['unsigned'])
        self.assertTrue(timeline[-2]['content']['body'], {})
        self.assertEqual(timeline[-3]['event_id'], normal_msg_id)
        self.assertEqual(timeline[-3]['unsigned']['redacted_by'], redaction_id)
        self.assertEqual(timeline[-3]['content'], {})

    def test_redact_nonexistent_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        b = self.helper.send(room_id=self.room_id, tok=self.other_access_token)
        msg_id = b['event_id']
        b = self._redact_event(self.other_access_token, self.room_id, msg_id)
        redaction_id = b['event_id']
        self._redact_event(self.mod_access_token, self.room_id, '$zzz')
        self._redact_event(self.other_access_token, self.room_id, '$zzz', expect_code=404)
        timeline = self._sync_room_timeline(self.other_access_token, self.room_id)
        self.assertEqual(timeline[-1]['event_id'], redaction_id)
        self.assertEqual(timeline[-1]['redacts'], msg_id)
        self.assertEqual(timeline[-2]['event_id'], msg_id)
        self.assertEqual(timeline[-2]['unsigned']['redacted_by'], redaction_id)
        self.assertEqual(timeline[-2]['content'], {})

    def test_redact_create_event(self) -> None:
        if False:
            i = 10
            return i + 15
        b = self.helper.send(room_id=self.room_id, tok=self.mod_access_token)
        msg_id = b['event_id']
        self._redact_event(self.mod_access_token, self.room_id, msg_id)
        timeline = self._sync_room_timeline(self.other_access_token, self.room_id)
        create_event_id = timeline[0]['event_id']
        self._redact_event(self.mod_access_token, self.room_id, create_event_id, expect_code=403)
        self._redact_event(self.other_access_token, self.room_id, create_event_id, expect_code=403)

    def test_redact_event_as_moderator_ratelimit(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the correct ratelimiting is applied to redactions'
        message_ids = []
        for _ in range(20):
            b = self.helper.send(room_id=self.room_id, tok=self.other_access_token)
            message_ids.append(b['event_id'])
            self.reactor.advance(10)
        for msg_id in message_ids:
            self._redact_event(self.mod_access_token, self.room_id, msg_id)

    @override_config({'experimental_features': {'msc3912_enabled': True}})
    def test_redact_relations_with_types(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that we can redact the relations of an event of specific types\n        at the same time as the event itself.\n        '
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'hello'}, tok=self.mod_access_token)
        root_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'body': ' * hello world', 'm.new_content': {'body': 'hello world', 'msgtype': 'm.text'}, 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.REPLACE}, 'msgtype': 'm.text'}, tok=self.mod_access_token)
        edit_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'message 1', 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.THREAD}}, tok=self.mod_access_token)
        threaded_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Reaction, content={'m.relates_to': {'rel_type': RelationTypes.ANNOTATION, 'event_id': root_event_id, 'key': '👍'}}, tok=self.mod_access_token)
        reaction_event_id = res['event_id']
        self._redact_event(self.mod_access_token, self.room_id, root_event_id, with_relations=[RelationTypes.REPLACE, RelationTypes.THREAD])
        event_dict = self.helper.get_event(self.room_id, root_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, edit_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, threaded_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, reaction_event_id, self.mod_access_token)
        self.assertNotIn('redacted_because', event_dict, event_dict)

    @override_config({'experimental_features': {'msc3912_enabled': True}})
    def test_redact_all_relations(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that we can redact all the relations of an event at the same time as the\n        event itself.\n        '
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'hello'}, tok=self.mod_access_token)
        root_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'body': ' * hello world', 'm.new_content': {'body': 'hello world', 'msgtype': 'm.text'}, 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.REPLACE}, 'msgtype': 'm.text'}, tok=self.mod_access_token)
        edit_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'message 1', 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.THREAD}}, tok=self.mod_access_token)
        threaded_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Reaction, content={'m.relates_to': {'rel_type': RelationTypes.ANNOTATION, 'event_id': root_event_id, 'key': '👍'}}, tok=self.mod_access_token)
        reaction_event_id = res['event_id']
        self._redact_event(self.mod_access_token, self.room_id, root_event_id, with_relations=['*'])
        event_dict = self.helper.get_event(self.room_id, root_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, edit_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, threaded_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, reaction_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)

    @override_config({'experimental_features': {'msc3912_enabled': True}})
    def test_redact_relations_no_perms(self) -> None:
        if False:
            print('Hello World!')
        'Tests that, when redacting a message along with its relations, if not all\n        the related messages can be redacted because of insufficient permissions, the\n        server still redacts all the ones that can be.\n        '
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'root'}, tok=self.other_access_token)
        root_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'message 1', 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.THREAD}}, tok=self.mod_access_token)
        first_threaded_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'message 2', 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.THREAD}}, tok=self.other_access_token)
        second_threaded_event_id = res['event_id']
        self._redact_event(self.other_access_token, self.room_id, root_event_id, with_relations=[RelationTypes.THREAD])
        event_dict = self.helper.get_event(self.room_id, root_event_id, self.other_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, second_threaded_event_id, self.other_access_token)
        self.assertIn('redacted_because', event_dict, event_dict)
        event_dict = self.helper.get_event(self.room_id, first_threaded_event_id, self.other_access_token)
        self.assertIn('body', event_dict['content'], event_dict)
        self.assertEqual('message 1', event_dict['content']['body'])

    @override_config({'experimental_features': {'msc3912_enabled': True}})
    def test_redact_relations_txn_id_reuse(self) -> None:
        if False:
            return 10
        'Tests that redacting a message using a transaction ID, then reusing the same\n        transaction ID but providing an additional list of relations to redact, is\n        effectively a no-op.\n        '
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'root'}, tok=self.mod_access_token)
        root_event_id = res['event_id']
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': "I'm in a thread!", 'm.relates_to': {'event_id': root_event_id, 'rel_type': RelationTypes.THREAD}}, tok=self.mod_access_token)
        threaded_event_id = res['event_id']
        channel = self.make_request(method='PUT', path=f'/rooms/{self.room_id}/redact/{root_event_id}/foo', content={}, access_token=self.mod_access_token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request(method='PUT', path=f'/rooms/{self.room_id}/redact/{root_event_id}/foo', content={'org.matrix.msc3912.with_relations': [RelationTypes.THREAD]}, access_token=self.mod_access_token)
        self.assertEqual(channel.code, 200)
        event_dict = self.helper.get_event(self.room_id, root_event_id, self.mod_access_token)
        self.assertIn('redacted_because', event_dict)
        event_dict = self.helper.get_event(self.room_id, threaded_event_id, self.mod_access_token)
        self.assertIn('body', event_dict['content'], event_dict)
        self.assertEqual("I'm in a thread!", event_dict['content']['body'])

    @parameterized.expand([(RoomVersions.V10, False, False), (RoomVersions.V11, True, True), (RoomVersions.V11, False, True)])
    def test_redaction_content(self, room_version: RoomVersion, include_content: bool, expect_content: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Room version 11 moved the redacts property to the content.\n\n        Ensure that the event gets created properly and that the Client-Server\n        API servers the proper backwards-compatible version.\n        '
        room_id = self.helper.create_room_as(self.mod_user_id, tok=self.mod_access_token, room_version=room_version.identifier)
        b = self.helper.send(room_id=room_id, tok=self.mod_access_token)
        event_id = b['event_id']
        if include_content:
            self._redact_event(self.mod_access_token, room_id, event_id, expect_code=400, content={'redacts': 'foo'})
        result = self._redact_event(self.mod_access_token, room_id, event_id, content={'redacts': event_id} if include_content else {})
        redaction_event_id = result['event_id']
        timeline = self._sync_room_timeline(self.mod_access_token, room_id)
        redact_event = timeline[-1]
        self.assertEqual(redact_event['type'], EventTypes.Redaction)
        self.assertEqual(redact_event['content']['redacts'], event_id)
        self.assertEqual(redact_event['redacts'], event_id)

        def get_event(txn: LoggingTransaction) -> JsonDict:
            if False:
                return 10
            return db_to_json(main_datastore._fetch_event_rows(txn, [redaction_event_id])[redaction_event_id].json)
        main_datastore = self.hs.get_datastores().main
        event_json = self.get_success(main_datastore.db_pool.runInteraction('get_event', get_event))
        self.assertEqual(event_json['type'], EventTypes.Redaction)
        if expect_content:
            self.assertNotIn('redacts', event_json)
            self.assertEqual(event_json['content']['redacts'], event_id)
        else:
            self.assertEqual(event_json['redacts'], event_id)
            self.assertNotIn('redacts', event_json['content'])