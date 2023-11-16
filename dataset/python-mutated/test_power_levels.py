from http import HTTPStatus
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.errors import Codes
from synapse.events.utils import CANONICALJSON_MAX_INT, CANONICALJSON_MIN_INT
from synapse.rest import admin
from synapse.rest.client import login, room, sync
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class PowerLevelsTestCase(HomeserverTestCase):
    """Tests that power levels are enforced in various situations"""
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets, sync.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        config = self.default_config()
        return self.setup_test_homeserver(config=config)

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.admin_user_id = self.register_user('admin', 'pass')
        self.admin_access_token = self.login('admin', 'pass')
        self.mod_user_id = self.register_user('mod', 'pass')
        self.mod_access_token = self.login('mod', 'pass')
        self.user_user_id = self.register_user('user', 'pass')
        self.user_access_token = self.login('user', 'pass')
        self.room_id = self.helper.create_room_as(self.admin_user_id, tok=self.admin_access_token)
        self.helper.invite(room=self.room_id, src=self.admin_user_id, tok=self.admin_access_token, targ=self.mod_user_id)
        self.helper.invite(room=self.room_id, src=self.admin_user_id, tok=self.admin_access_token, targ=self.user_user_id)
        self.helper.join(room=self.room_id, user=self.mod_user_id, tok=self.mod_access_token)
        self.helper.join(room=self.room_id, user=self.user_user_id, tok=self.user_access_token)
        room_power_levels = self.helper.get_state(self.room_id, 'm.room.power_levels', tok=self.admin_access_token)
        room_power_levels['users'].update({self.mod_user_id: 50})
        self.helper.send_state(self.room_id, 'm.room.power_levels', room_power_levels, tok=self.admin_access_token)

    def test_non_admins_cannot_enable_room_encryption(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.helper.send_state(self.room_id, 'm.room.encryption', {'algorithm': 'm.megolm.v1.aes-sha2'}, tok=self.mod_access_token, expect_code=403)
        self.helper.send_state(self.room_id, 'm.room.encryption', {'algorithm': 'm.megolm.v1.aes-sha2'}, tok=self.user_access_token, expect_code=HTTPStatus.FORBIDDEN)

    def test_non_admins_cannot_send_server_acl(self) -> None:
        if False:
            i = 10
            return i + 15
        self.helper.send_state(self.room_id, 'm.room.server_acl', {'allow': ['*'], 'allow_ip_literals': False, 'deny': ['*.evil.com', 'evil.com']}, tok=self.mod_access_token, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.send_state(self.room_id, 'm.room.server_acl', {'allow': ['*'], 'allow_ip_literals': False, 'deny': ['*.evil.com', 'evil.com']}, tok=self.user_access_token, expect_code=HTTPStatus.FORBIDDEN)

    def test_non_admins_cannot_tombstone_room(self) -> None:
        if False:
            while True:
                i = 10
        self.upgraded_room_id = self.helper.create_room_as(self.admin_user_id, tok=self.admin_access_token)
        self.helper.send_state(self.room_id, 'm.room.tombstone', {'body': 'This room has been replaced', 'replacement_room': self.upgraded_room_id}, tok=self.mod_access_token, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.send_state(self.room_id, 'm.room.tombstone', {'body': 'This room has been replaced', 'replacement_room': self.upgraded_room_id}, tok=self.user_access_token, expect_code=403)

    def test_admins_can_enable_room_encryption(self) -> None:
        if False:
            print('Hello World!')
        self.helper.send_state(self.room_id, 'm.room.encryption', {'algorithm': 'm.megolm.v1.aes-sha2'}, tok=self.admin_access_token, expect_code=HTTPStatus.OK)

    def test_admins_can_send_server_acl(self) -> None:
        if False:
            i = 10
            return i + 15
        self.helper.send_state(self.room_id, 'm.room.server_acl', {'allow': ['*'], 'allow_ip_literals': False, 'deny': ['*.evil.com', 'evil.com']}, tok=self.admin_access_token, expect_code=HTTPStatus.OK)

    def test_admins_can_tombstone_room(self) -> None:
        if False:
            while True:
                i = 10
        self.upgraded_room_id = self.helper.create_room_as(self.admin_user_id, tok=self.admin_access_token)
        self.helper.send_state(self.room_id, 'm.room.tombstone', {'body': 'This room has been replaced', 'replacement_room': self.upgraded_room_id}, tok=self.admin_access_token, expect_code=HTTPStatus.OK)

    def test_cannot_set_string_power_levels(self) -> None:
        if False:
            while True:
                i = 10
        room_power_levels = self.helper.get_state(self.room_id, 'm.room.power_levels', tok=self.admin_access_token)
        room_power_levels['users'].update({self.user_user_id: '0'})
        body = self.helper.send_state(self.room_id, 'm.room.power_levels', room_power_levels, tok=self.admin_access_token, expect_code=HTTPStatus.BAD_REQUEST)
        self.assertEqual(body['errcode'], Codes.BAD_JSON, body)

    def test_cannot_set_unsafe_large_power_levels(self) -> None:
        if False:
            i = 10
            return i + 15
        room_power_levels = self.helper.get_state(self.room_id, 'm.room.power_levels', tok=self.admin_access_token)
        room_power_levels['users'].update({self.user_user_id: CANONICALJSON_MAX_INT + 1})
        body = self.helper.send_state(self.room_id, 'm.room.power_levels', room_power_levels, tok=self.admin_access_token, expect_code=HTTPStatus.BAD_REQUEST)
        self.assertEqual(body['errcode'], Codes.BAD_JSON, body)

    def test_cannot_set_unsafe_small_power_levels(self) -> None:
        if False:
            return 10
        room_power_levels = self.helper.get_state(self.room_id, 'm.room.power_levels', tok=self.admin_access_token)
        room_power_levels['users'].update({self.user_user_id: CANONICALJSON_MIN_INT - 1})
        body = self.helper.send_state(self.room_id, 'm.room.power_levels', room_power_levels, tok=self.admin_access_token, expect_code=HTTPStatus.BAD_REQUEST)
        self.assertEqual(body['errcode'], Codes.BAD_JSON, body)