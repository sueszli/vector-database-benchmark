"""A federation sender that forwards things to be sent across replication to
a worker process.

It assumes there is a single worker process feeding off of it.

Each row in the replication stream consists of a type and some json, where the
types indicate whether they are presence, or edus, etc.

Ephemeral or non-event data are queued up in-memory. When the worker requests
updates since a particular point, all in-memory data since before that point is
dropped. We also expire things in the queue after 5 minutes, to ensure that a
dead worker doesn't cause the queues to grow limitlessly.

Events are replicated via a separate events stream.
"""
import logging
from typing import TYPE_CHECKING, Dict, Hashable, Iterable, List, Optional, Sized, Tuple, Type
import attr
from sortedcontainers import SortedDict
from synapse.api.presence import UserPresenceState
from synapse.federation.sender import AbstractFederationSender, FederationSender
from synapse.metrics import LaterGauge
from synapse.replication.tcp.streams.federation import FederationStream
from synapse.types import JsonDict, ReadReceipt, RoomStreamToken, StrCollection
from synapse.util.metrics import Measure
from .units import Edu
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class FederationRemoteSendQueue(AbstractFederationSender):
    """A drop in replacement for FederationSender"""

    def __init__(self, hs: 'HomeServer'):
        if False:
            return 10
        self.server_name = hs.hostname
        self.clock = hs.get_clock()
        self.notifier = hs.get_notifier()
        self.is_mine_id = hs.is_mine_id
        self.is_mine_server_name = hs.is_mine_server_name
        self._sender_instances = hs.config.worker.federation_shard_config.instances
        self._sender_positions: Dict[str, int] = {}
        self.presence_map: Dict[str, UserPresenceState] = {}
        self.presence_destinations: SortedDict[int, Tuple[str, Iterable[str]]] = SortedDict()
        self.keyed_edu: Dict[Tuple[str, tuple], Edu] = {}
        self.keyed_edu_changed: SortedDict[int, Tuple[str, tuple]] = SortedDict()
        self.edus: SortedDict[int, Edu] = SortedDict()
        self.pos = 1
        self.pos_time: SortedDict[int, int] = SortedDict()

        def register(name: str, queue: Sized) -> None:
            if False:
                return 10
            LaterGauge('synapse_federation_send_queue_%s_size' % (queue_name,), '', [], lambda : len(queue))
        for queue_name in ['presence_map', 'keyed_edu', 'keyed_edu_changed', 'edus', 'pos_time', 'presence_destinations']:
            register(queue_name, getattr(self, queue_name))
        self.clock.looping_call(self._clear_queue, 30 * 1000)

    def _next_pos(self) -> int:
        if False:
            i = 10
            return i + 15
        pos = self.pos
        self.pos += 1
        self.pos_time[self.clock.time_msec()] = pos
        return pos

    def _clear_queue(self) -> None:
        if False:
            i = 10
            return i + 15
        'Clear the queues for anything older than N minutes'
        FIVE_MINUTES_AGO = 5 * 60 * 1000
        now = self.clock.time_msec()
        keys = self.pos_time.keys()
        time = self.pos_time.bisect_left(now - FIVE_MINUTES_AGO)
        if not keys[:time]:
            return
        position_to_delete = max(keys[:time])
        for key in keys[:time]:
            del self.pos_time[key]
        self._clear_queue_before_pos(position_to_delete)

    def _clear_queue_before_pos(self, position_to_delete: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear all the queues from before a given position'
        with Measure(self.clock, 'send_queue._clear'):
            keys = self.presence_destinations.keys()
            i = self.presence_destinations.bisect_left(position_to_delete)
            for key in keys[:i]:
                del self.presence_destinations[key]
            user_ids = {user_id for (user_id, _) in self.presence_destinations.values()}
            to_del = [user_id for user_id in self.presence_map if user_id not in user_ids]
            for user_id in to_del:
                del self.presence_map[user_id]
            keys = self.keyed_edu_changed.keys()
            i = self.keyed_edu_changed.bisect_left(position_to_delete)
            for key in keys[:i]:
                del self.keyed_edu_changed[key]
            live_keys = set()
            for edu_key in self.keyed_edu_changed.values():
                live_keys.add(edu_key)
            keys_to_del = [edu_key for edu_key in self.keyed_edu if edu_key not in live_keys]
            for edu_key in keys_to_del:
                del self.keyed_edu[edu_key]
            keys = self.edus.keys()
            i = self.edus.bisect_left(position_to_delete)
            for key in keys[:i]:
                del self.edus[key]

    def notify_new_events(self, max_token: RoomStreamToken) -> None:
        if False:
            i = 10
            return i + 15
        'As per FederationSender'
        raise NotImplementedError()

    def build_and_send_edu(self, destination: str, edu_type: str, content: JsonDict, key: Optional[Hashable]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'As per FederationSender'
        if self.is_mine_server_name(destination):
            logger.info('Not sending EDU to ourselves')
            return
        pos = self._next_pos()
        edu = Edu(origin=self.server_name, destination=destination, edu_type=edu_type, content=content)
        if key:
            assert isinstance(key, tuple)
            self.keyed_edu[destination, key] = edu
            self.keyed_edu_changed[pos] = (destination, key)
        else:
            self.edus[pos] = edu
        self.notifier.on_new_replication_data()

    async def send_read_receipt(self, receipt: ReadReceipt) -> None:
        """As per FederationSender

        Args:
            receipt:
        """

    async def send_presence_to_destinations(self, states: Iterable[UserPresenceState], destinations: Iterable[str]) -> None:
        """As per FederationSender

        Args:
            states
            destinations
        """
        for state in states:
            pos = self._next_pos()
            self.presence_map.update({state.user_id: state for state in states})
            self.presence_destinations[pos] = (state.user_id, destinations)
        self.notifier.on_new_replication_data()

    async def send_device_messages(self, destinations: StrCollection, immediate: bool=True) -> None:
        """As per FederationSender"""

    def wake_destination(self, server: str) -> None:
        if False:
            print('Hello World!')
        pass

    def get_current_token(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.pos - 1

    def federation_ack(self, instance_name: str, token: int) -> None:
        if False:
            while True:
                i = 10
        if self._sender_instances:
            self._sender_positions[instance_name] = token
            token = min(self._sender_positions.values())
        self._clear_queue_before_pos(token)

    async def get_replication_rows(self, instance_name: str, from_token: int, to_token: int, target_row_count: int) -> Tuple[List[Tuple[int, Tuple]], int, bool]:
        """Get rows to be sent over federation between the two tokens

        Args:
            instance_name: the name of the current process
            from_token: the previous stream token: the starting point for fetching the
                updates
            to_token: the new stream token: the point to get updates up to
            target_row_count: a target for the number of rows to be returned.

        Returns: a triplet `(updates, new_last_token, limited)`, where:
           * `updates` is a list of `(token, row)` entries.
           * `new_last_token` is the new position in stream.
           * `limited` is whether there are more updates to fetch.
        """
        if from_token > self.pos:
            from_token = -1
        rows: List[Tuple[int, BaseFederationRow]] = []
        i = self.presence_destinations.bisect_right(from_token)
        j = self.presence_destinations.bisect_right(to_token) + 1
        for (pos, (user_id, dests)) in self.presence_destinations.items()[i:j]:
            rows.append((pos, PresenceDestinationsRow(state=self.presence_map[user_id], destinations=list(dests))))
        i = self.keyed_edu_changed.bisect_right(from_token)
        j = self.keyed_edu_changed.bisect_right(to_token) + 1
        keyed_edus = {v: k for (k, v) in self.keyed_edu_changed.items()[i:j]}
        for ((destination, edu_key), pos) in keyed_edus.items():
            rows.append((pos, KeyedEduRow(key=edu_key, edu=self.keyed_edu[destination, edu_key])))
        i = self.edus.bisect_right(from_token)
        j = self.edus.bisect_right(to_token) + 1
        edus = self.edus.items()[i:j]
        for (pos, edu) in edus:
            rows.append((pos, EduRow(edu)))
        rows.sort()
        return ([(pos, (row.TypeId, row.to_data())) for (pos, row) in rows], to_token, False)

class BaseFederationRow:
    """Base class for rows to be sent in the federation stream.

    Specifies how to identify, serialize and deserialize the different types.
    """
    TypeId = ''

    @staticmethod
    def from_data(data: JsonDict) -> 'BaseFederationRow':
        if False:
            for i in range(10):
                print('nop')
        'Parse the data from the federation stream into a row.\n\n        Args:\n            data: The value of ``data`` from FederationStreamRow.data, type\n                depends on the type of stream\n        '
        raise NotImplementedError()

    def to_data(self) -> JsonDict:
        if False:
            print('Hello World!')
        'Serialize this row to be sent over the federation stream.\n\n        Returns:\n            The value to be sent in FederationStreamRow.data. The type depends\n            on the type of stream.\n        '
        raise NotImplementedError()

    def add_to_buffer(self, buff: 'ParsedFederationStreamData') -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add this row to the appropriate field in the buffer ready for this\n        to be sent over federation.\n\n        We use a buffer so that we can batch up events that have come in at\n        the same time and send them all at once.\n\n        Args:\n            buff (BufferedToSend)\n        '
        raise NotImplementedError()

@attr.s(slots=True, frozen=True, auto_attribs=True)
class PresenceDestinationsRow(BaseFederationRow):
    state: UserPresenceState
    destinations: List[str]
    TypeId = 'pd'

    @staticmethod
    def from_data(data: JsonDict) -> 'PresenceDestinationsRow':
        if False:
            return 10
        return PresenceDestinationsRow(state=UserPresenceState(**data['state']), destinations=data['dests'])

    def to_data(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        return {'state': self.state.as_dict(), 'dests': self.destinations}

    def add_to_buffer(self, buff: 'ParsedFederationStreamData') -> None:
        if False:
            i = 10
            return i + 15
        buff.presence_destinations.append((self.state, self.destinations))

@attr.s(slots=True, frozen=True, auto_attribs=True)
class KeyedEduRow(BaseFederationRow):
    """Streams EDUs that have an associated key that is ued to clobber. For example,
    typing EDUs clobber based on room_id.
    """
    key: Tuple[str, ...]
    edu: Edu
    TypeId = 'k'

    @staticmethod
    def from_data(data: JsonDict) -> 'KeyedEduRow':
        if False:
            i = 10
            return i + 15
        return KeyedEduRow(key=tuple(data['key']), edu=Edu(**data['edu']))

    def to_data(self) -> JsonDict:
        if False:
            print('Hello World!')
        return {'key': self.key, 'edu': self.edu.get_internal_dict()}

    def add_to_buffer(self, buff: 'ParsedFederationStreamData') -> None:
        if False:
            i = 10
            return i + 15
        buff.keyed_edus.setdefault(self.edu.destination, {})[self.key] = self.edu

@attr.s(slots=True, frozen=True, auto_attribs=True)
class EduRow(BaseFederationRow):
    """Streams EDUs that don't have keys. See KeyedEduRow"""
    edu: Edu
    TypeId = 'e'

    @staticmethod
    def from_data(data: JsonDict) -> 'EduRow':
        if False:
            return 10
        return EduRow(Edu(**data))

    def to_data(self) -> JsonDict:
        if False:
            print('Hello World!')
        return self.edu.get_internal_dict()

    def add_to_buffer(self, buff: 'ParsedFederationStreamData') -> None:
        if False:
            print('Hello World!')
        buff.edus.setdefault(self.edu.destination, []).append(self.edu)
_rowtypes: Tuple[Type[BaseFederationRow], ...] = (PresenceDestinationsRow, KeyedEduRow, EduRow)
TypeToRow = {Row.TypeId: Row for Row in _rowtypes}

@attr.s(slots=True, frozen=True, auto_attribs=True)
class ParsedFederationStreamData:
    presence_destinations: List[Tuple[UserPresenceState, List[str]]]
    keyed_edus: Dict[str, Dict[Tuple[str, ...], Edu]]
    edus: Dict[str, List[Edu]]

async def process_rows_for_federation(transaction_queue: FederationSender, rows: List[FederationStream.FederationStreamRow]) -> None:
    """Parse a list of rows from the federation stream and put them in the
    transaction queue ready for sending to the relevant homeservers.

    Args:
        transaction_queue
        rows
    """
    buff = ParsedFederationStreamData(presence_destinations=[], keyed_edus={}, edus={})
    for row in rows:
        if row.type not in TypeToRow:
            logger.error('Unrecognized federation row type %r', row.type)
            continue
        RowType = TypeToRow[row.type]
        parsed_row = RowType.from_data(row.data)
        parsed_row.add_to_buffer(buff)
    for (state, destinations) in buff.presence_destinations:
        await transaction_queue.send_presence_to_destinations(states=[state], destinations=destinations)
    for edu_map in buff.keyed_edus.values():
        for (key, edu) in edu_map.items():
            transaction_queue.send_edu(edu, key)
    for edu_list in buff.edus.values():
        for edu in edu_list:
            transaction_queue.send_edu(edu, None)