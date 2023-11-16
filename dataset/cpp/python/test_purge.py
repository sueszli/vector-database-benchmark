# Copyright 2018 New Vector Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from twisted.test.proto_helpers import MemoryReactor

from synapse.api.errors import NotFoundError, SynapseError
from synapse.rest.client import room
from synapse.server import HomeServer
from synapse.util import Clock

from tests.unittest import HomeserverTestCase


class PurgeTests(HomeserverTestCase):
    user_id = "@red:server"
    servlets = [room.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        hs = self.setup_test_homeserver("server")
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        self.room_id = self.helper.create_room_as(self.user_id)

        self.store = hs.get_datastores().main
        self._storage_controllers = self.hs.get_storage_controllers()

    def test_purge_history(self) -> None:
        """
        Purging a room history will delete everything before the topological point.
        """
        # Send four messages to the room
        first = self.helper.send(self.room_id, body="test1")
        second = self.helper.send(self.room_id, body="test2")
        third = self.helper.send(self.room_id, body="test3")
        last = self.helper.send(self.room_id, body="test4")

        # Get the topological token
        token = self.get_success(
            self.store.get_topological_token_for_event(last["event_id"])
        )
        token_str = self.get_success(token.to_string(self.hs.get_datastores().main))

        # Purge everything before this topological token
        self.get_success(
            self._storage_controllers.purge_events.purge_history(
                self.room_id, token_str, True
            )
        )

        # 1-3 should fail and last will succeed, meaning that 1-3 are deleted
        # and last is not.
        self.get_failure(self.store.get_event(first["event_id"]), NotFoundError)
        self.get_failure(self.store.get_event(second["event_id"]), NotFoundError)
        self.get_failure(self.store.get_event(third["event_id"]), NotFoundError)
        self.get_success(self.store.get_event(last["event_id"]))

    def test_purge_history_wont_delete_extrems(self) -> None:
        """
        Purging a room history will delete everything before the topological point.
        """
        # Send four messages to the room
        first = self.helper.send(self.room_id, body="test1")
        second = self.helper.send(self.room_id, body="test2")
        third = self.helper.send(self.room_id, body="test3")
        last = self.helper.send(self.room_id, body="test4")

        # Set the topological token higher than it should be
        token = self.get_success(
            self.store.get_topological_token_for_event(last["event_id"])
        )
        assert token.topological is not None
        event = f"t{token.topological + 1}-{token.stream + 1}"

        # Purge everything before this topological token
        f = self.get_failure(
            self._storage_controllers.purge_events.purge_history(
                self.room_id, event, True
            ),
            SynapseError,
        )
        self.assertIn("greater than forward", f.value.args[0])

        # Try and get the events
        self.get_success(self.store.get_event(first["event_id"]))
        self.get_success(self.store.get_event(second["event_id"]))
        self.get_success(self.store.get_event(third["event_id"]))
        self.get_success(self.store.get_event(last["event_id"]))

    def test_purge_room(self) -> None:
        """
        Purging a room will delete everything about it.
        """
        # Send four messages to the room
        first = self.helper.send(self.room_id, body="test1")

        # Get the current room state.
        create_event = self.get_success(
            self._storage_controllers.state.get_current_state_event(
                self.room_id, "m.room.create", ""
            )
        )
        assert create_event is not None

        # Purge everything before this topological token
        self.get_success(
            self._storage_controllers.purge_events.purge_room(self.room_id)
        )

        # The events aren't found.
        self.store._invalidate_local_get_event_cache(create_event.event_id)
        self.get_failure(self.store.get_event(create_event.event_id), NotFoundError)
        self.get_failure(self.store.get_event(first["event_id"]), NotFoundError)
