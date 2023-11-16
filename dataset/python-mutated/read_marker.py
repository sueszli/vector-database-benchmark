import logging
from typing import TYPE_CHECKING
from synapse.api.constants import ReceiptTypes
from synapse.api.errors import SynapseError
from synapse.util.async_helpers import Linearizer
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class ReadMarkerHandler:

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        self.store = hs.get_datastores().main
        self.account_data_handler = hs.get_account_data_handler()
        self.read_marker_linearizer = Linearizer(name='read_marker')

    async def received_client_read_marker(self, room_id: str, user_id: str, event_id: str) -> None:
        """Updates the read marker for a given user in a given room if the event ID given
        is ahead in the stream relative to the current read marker.

        This uses a notifier to indicate that account data should be sent down /sync if
        the read marker has changed.
        """
        async with self.read_marker_linearizer.queue((room_id, user_id)):
            existing_read_marker = await self.store.get_account_data_for_room_and_type(user_id, room_id, ReceiptTypes.FULLY_READ)
            should_update = True
            event_ordering = await self.store.get_event_ordering(event_id)
            if existing_read_marker:
                try:
                    old_event_ordering = await self.store.get_event_ordering(existing_read_marker['event_id'])
                except SynapseError:
                    pass
                else:
                    should_update = event_ordering > old_event_ordering
            if should_update:
                content = {'event_id': event_id}
                await self.account_data_handler.add_account_data_to_room(user_id, room_id, ReceiptTypes.FULLY_READ, content)