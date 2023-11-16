from typing import List, Optional, Tuple, cast
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import Membership
from synapse.rest.admin import register_servlets_for_client_rest_resource
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.types import UserID, create_requester
from synapse.util import Clock
from tests import unittest
from tests.server import TestHomeServer
from tests.test_utils import event_injection

class RoomMemberStoreTestCase(unittest.HomeserverTestCase):
    servlets = [login.register_servlets, register_servlets_for_client_rest_resource, room.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: TestHomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.u_alice = self.register_user('alice', 'pass')
        self.t_alice = self.login('alice', 'pass')
        self.u_bob = self.register_user('bob', 'pass')
        self.u_charlie = UserID.from_string('@charlie:elsewhere')

    def test_one_member(self) -> None:
        if False:
            print('Hello World!')
        self.room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        rooms_for_user = self.get_success(self.store.get_rooms_for_local_user_where_membership_is(self.u_alice, [Membership.JOIN]))
        self.assertEqual([self.room], [m.room_id for m in rooms_for_user])

    def test_count_known_servers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        _count_known_servers will calculate how many servers are in a room.\n        '
        self.room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        self.inject_room_member(self.room, self.u_bob, Membership.JOIN)
        self.inject_room_member(self.room, self.u_charlie.to_string(), Membership.JOIN)
        servers = self.get_success(self.store._count_known_servers())
        self.assertEqual(servers, 2)

    def test_count_known_servers_stat_counter_disabled(self) -> None:
        if False:
            while True:
                i = 10
        '\n        If enabled, the metrics for how many servers are known will be counted.\n        '
        self.assertTrue('_known_servers_count' not in self.store.__dict__.keys())
        self.room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        self.inject_room_member(self.room, self.u_bob, Membership.JOIN)
        self.inject_room_member(self.room, self.u_charlie.to_string(), Membership.JOIN)
        self.pump()
        self.assertTrue('_known_servers_count' not in self.store.__dict__.keys())

    @unittest.override_config({'enable_metrics': True, 'metrics_flags': {'known_servers': True}})
    def test_count_known_servers_stat_counter_enabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        If enabled, the metrics for how many servers are known will be counted.\n        '
        self.assertEqual(self.store._known_servers_count, 1)
        self.pump()
        self.assertEqual(self.store._known_servers_count, 1)
        self.room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        self.inject_room_member(self.room, self.u_bob, Membership.JOIN)
        self.inject_room_member(self.room, self.u_charlie.to_string(), Membership.JOIN)
        self.pump(1)
        self.assertEqual(self.store._known_servers_count, 2)

    def test__null_byte_in_display_name_properly_handled(self) -> None:
        if False:
            return 10
        room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        res = cast(List[Tuple[Optional[str], str]], self.get_success(self.store.db_pool.simple_select_list('room_memberships', {'user_id': '@alice:test'}, ['display_name', 'event_id'])))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], 'alice')
        event_id = res[0][1]
        new_profile = {'displayname': 'ali\x00ce'}
        self.helper.change_membership(room, self.u_alice, self.u_alice, 'join', extra_data=new_profile, tok=self.t_alice)
        res2 = cast(List[Tuple[Optional[str], str]], self.get_success(self.store.db_pool.simple_select_list('room_memberships', {'user_id': '@alice:test'}, ['display_name', 'event_id'])))
        self.assertEqual(len(res2), 2)
        row = [row for row in res2 if row[1] != event_id]
        self.assertIsNone(row[0][0])

    def test_room_is_locally_forgotten(self) -> None:
        if False:
            print('Hello World!')
        'Test that when the last local user has forgotten a room it is known as forgotten.'
        self.room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        self.get_success(event_injection.inject_member_event(self.hs, self.room, self.u_bob, 'join'))
        self.get_success(event_injection.inject_member_event(self.hs, self.room, self.u_charlie.to_string(), 'join'))
        self.assertFalse(self.get_success(self.store.is_locally_forgotten_room(self.room)))
        self.get_success(event_injection.inject_member_event(self.hs, self.room, self.u_alice, 'leave'))
        self.get_success(event_injection.inject_member_event(self.hs, self.room, self.u_bob, 'leave'))
        self.assertFalse(self.get_success(self.store.is_locally_forgotten_room(self.room)))
        self.get_success(self.store.forget(self.u_alice, self.room))
        self.assertFalse(self.get_success(self.store.is_locally_forgotten_room(self.room)))
        self.get_success(self.store.forget(self.u_bob, self.room))
        self.assertTrue(self.get_success(self.store.is_locally_forgotten_room(self.room)))

    def test_join_locally_forgotten_room(self) -> None:
        if False:
            while True:
                i = 10
        'Tests if a user joins a forgotten room the room is not forgotten anymore.'
        self.room = self.helper.create_room_as(self.u_alice, tok=self.t_alice)
        self.assertFalse(self.get_success(self.store.is_locally_forgotten_room(self.room)))
        self.get_success(event_injection.inject_member_event(self.hs, self.room, self.u_alice, 'leave'))
        self.get_success(self.store.forget(self.u_alice, self.room))
        self.assertTrue(self.get_success(self.store.is_locally_forgotten_room(self.room)))
        self.get_success(event_injection.inject_member_event(self.hs, self.room, self.u_alice, 'join'))
        self.assertFalse(self.get_success(self.store.is_locally_forgotten_room(self.room)))

class CurrentStateMembershipUpdateTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.room_creator = hs.get_room_creation_handler()

    def test_can_rerun_update(self) -> None:
        if False:
            print('Hello World!')
        self.wait_for_background_updates()
        user = UserID('alice', 'test')
        requester = create_requester(user)
        self.get_success(self.room_creator.create_room(requester, {}))
        self.get_success(self.store.db_pool.simple_insert(table='background_updates', values={'update_name': 'current_state_events_membership', 'progress_json': '{}', 'depends_on': None}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()