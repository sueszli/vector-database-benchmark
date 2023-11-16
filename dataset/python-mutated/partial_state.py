from typing import TYPE_CHECKING
import attr
from synapse.replication.tcp.streams._base import _StreamFromIdGen
if TYPE_CHECKING:
    from synapse.server import HomeServer

@attr.s(slots=True, frozen=True, auto_attribs=True)
class UnPartialStatedRoomStreamRow:
    room_id: str

class UnPartialStatedRoomStream(_StreamFromIdGen):
    """
    Stream to notify about rooms becoming un-partial-stated;
    that is, when the background sync finishes such that we now have full state for
    the room.
    """
    NAME = 'un_partial_stated_room'
    ROW_TYPE = UnPartialStatedRoomStreamRow

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        store = hs.get_datastores().main
        super().__init__(hs.get_instance_name(), store.get_un_partial_stated_rooms_from_stream, store._un_partial_stated_rooms_stream_id_gen)

@attr.s(slots=True, frozen=True, auto_attribs=True)
class UnPartialStatedEventStreamRow:
    event_id: str
    rejection_status_changed: bool

class UnPartialStatedEventStream(_StreamFromIdGen):
    """
    Stream to notify about events becoming un-partial-stated.
    """
    NAME = 'un_partial_stated_event'
    ROW_TYPE = UnPartialStatedEventStreamRow

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        store = hs.get_datastores().main
        super().__init__(hs.get_instance_name(), store.get_un_partial_stated_events_from_stream, store._un_partial_stated_events_stream_id_gen)