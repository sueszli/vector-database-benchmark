"""Represent one server the driver is connected to."""
from __future__ import annotations
import time
import warnings
from typing import Any, Mapping, Optional
from bson import EPOCH_NAIVE
from bson.objectid import ObjectId
from pymongo.hello import Hello
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import ClusterTime, _Address

class ServerDescription:
    """Immutable representation of one server.

    :Parameters:
      - `address`: A (host, port) pair
      - `hello`: Optional Hello instance
      - `round_trip_time`: Optional float
      - `error`: Optional, the last error attempting to connect to the server
      - `round_trip_time`: Optional float, the min latency from the most recent samples
    """
    __slots__ = ('_address', '_server_type', '_all_hosts', '_tags', '_replica_set_name', '_primary', '_max_bson_size', '_max_message_size', '_max_write_batch_size', '_min_wire_version', '_max_wire_version', '_round_trip_time', '_min_round_trip_time', '_me', '_is_writable', '_is_readable', '_ls_timeout_minutes', '_error', '_set_version', '_election_id', '_cluster_time', '_last_write_date', '_last_update_time', '_topology_version')

    def __init__(self, address: _Address, hello: Optional[Hello]=None, round_trip_time: Optional[float]=None, error: Optional[Exception]=None, min_round_trip_time: float=0.0) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._address = address
        if not hello:
            hello = Hello({})
        self._server_type = hello.server_type
        self._all_hosts = hello.all_hosts
        self._tags = hello.tags
        self._replica_set_name = hello.replica_set_name
        self._primary = hello.primary
        self._max_bson_size = hello.max_bson_size
        self._max_message_size = hello.max_message_size
        self._max_write_batch_size = hello.max_write_batch_size
        self._min_wire_version = hello.min_wire_version
        self._max_wire_version = hello.max_wire_version
        self._set_version = hello.set_version
        self._election_id = hello.election_id
        self._cluster_time = hello.cluster_time
        self._is_writable = hello.is_writable
        self._is_readable = hello.is_readable
        self._ls_timeout_minutes = hello.logical_session_timeout_minutes
        self._round_trip_time = round_trip_time
        self._min_round_trip_time = min_round_trip_time
        self._me = hello.me
        self._last_update_time = time.monotonic()
        self._error = error
        self._topology_version = hello.topology_version
        if error:
            details = getattr(error, 'details', None)
            if isinstance(details, dict):
                self._topology_version = details.get('topologyVersion')
        self._last_write_date: Optional[float]
        if hello.last_write_date:
            delta = hello.last_write_date - EPOCH_NAIVE
            self._last_write_date = delta.total_seconds()
        else:
            self._last_write_date = None

    @property
    def address(self) -> _Address:
        if False:
            print('Hello World!')
        'The address (host, port) of this server.'
        return self._address

    @property
    def server_type(self) -> int:
        if False:
            print('Hello World!')
        'The type of this server.'
        return self._server_type

    @property
    def server_type_name(self) -> str:
        if False:
            return 10
        'The server type as a human readable string.\n\n        .. versionadded:: 3.4\n        '
        return SERVER_TYPE._fields[self._server_type]

    @property
    def all_hosts(self) -> set[tuple[str, int]]:
        if False:
            print('Hello World!')
        'List of hosts, passives, and arbiters known to this server.'
        return self._all_hosts

    @property
    def tags(self) -> Mapping[str, Any]:
        if False:
            return 10
        return self._tags

    @property
    def replica_set_name(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Replica set name or None.'
        return self._replica_set_name

    @property
    def primary(self) -> Optional[tuple[str, int]]:
        if False:
            for i in range(10):
                print('nop')
        "This server's opinion about who the primary is, or None."
        return self._primary

    @property
    def max_bson_size(self) -> int:
        if False:
            return 10
        return self._max_bson_size

    @property
    def max_message_size(self) -> int:
        if False:
            print('Hello World!')
        return self._max_message_size

    @property
    def max_write_batch_size(self) -> int:
        if False:
            print('Hello World!')
        return self._max_write_batch_size

    @property
    def min_wire_version(self) -> int:
        if False:
            return 10
        return self._min_wire_version

    @property
    def max_wire_version(self) -> int:
        if False:
            print('Hello World!')
        return self._max_wire_version

    @property
    def set_version(self) -> Optional[int]:
        if False:
            return 10
        return self._set_version

    @property
    def election_id(self) -> Optional[ObjectId]:
        if False:
            print('Hello World!')
        return self._election_id

    @property
    def cluster_time(self) -> Optional[ClusterTime]:
        if False:
            while True:
                i = 10
        return self._cluster_time

    @property
    def election_tuple(self) -> tuple[Optional[int], Optional[ObjectId]]:
        if False:
            while True:
                i = 10
        warnings.warn("'election_tuple' is deprecated, use  'set_version' and 'election_id' instead", DeprecationWarning, stacklevel=2)
        return (self._set_version, self._election_id)

    @property
    def me(self) -> Optional[tuple[str, int]]:
        if False:
            i = 10
            return i + 15
        return self._me

    @property
    def logical_session_timeout_minutes(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._ls_timeout_minutes

    @property
    def last_write_date(self) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        return self._last_write_date

    @property
    def last_update_time(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self._last_update_time

    @property
    def round_trip_time(self) -> Optional[float]:
        if False:
            while True:
                i = 10
        'The current average latency or None.'
        if self._address in self._host_to_round_trip_time:
            return self._host_to_round_trip_time[self._address]
        return self._round_trip_time

    @property
    def min_round_trip_time(self) -> float:
        if False:
            return 10
        'The min latency from the most recent samples.'
        return self._min_round_trip_time

    @property
    def error(self) -> Optional[Exception]:
        if False:
            i = 10
            return i + 15
        'The last error attempting to connect to the server, or None.'
        return self._error

    @property
    def is_writable(self) -> bool:
        if False:
            while True:
                i = 10
        return self._is_writable

    @property
    def is_readable(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._is_readable

    @property
    def mongos(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._server_type == SERVER_TYPE.Mongos

    @property
    def is_server_type_known(self) -> bool:
        if False:
            print('Hello World!')
        return self.server_type != SERVER_TYPE.Unknown

    @property
    def retryable_writes_supported(self) -> bool:
        if False:
            print('Hello World!')
        'Checks if this server supports retryable writes.'
        return self._ls_timeout_minutes is not None and self._server_type in (SERVER_TYPE.Mongos, SERVER_TYPE.RSPrimary) or self._server_type == SERVER_TYPE.LoadBalancer

    @property
    def retryable_reads_supported(self) -> bool:
        if False:
            while True:
                i = 10
        'Checks if this server supports retryable writes.'
        return self._max_wire_version >= 6

    @property
    def topology_version(self) -> Optional[Mapping[str, Any]]:
        if False:
            while True:
                i = 10
        return self._topology_version

    def to_unknown(self, error: Optional[Exception]=None) -> ServerDescription:
        if False:
            i = 10
            return i + 15
        unknown = ServerDescription(self.address, error=error)
        unknown._topology_version = self.topology_version
        return unknown

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if isinstance(other, ServerDescription):
            return self._address == other.address and self._server_type == other.server_type and (self._min_wire_version == other.min_wire_version) and (self._max_wire_version == other.max_wire_version) and (self._me == other.me) and (self._all_hosts == other.all_hosts) and (self._tags == other.tags) and (self._replica_set_name == other.replica_set_name) and (self._set_version == other.set_version) and (self._election_id == other.election_id) and (self._primary == other.primary) and (self._ls_timeout_minutes == other.logical_session_timeout_minutes) and (self._error == other.error)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return not self == other

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        errmsg = ''
        if self.error:
            errmsg = f', error={self.error!r}'
        return '<{} {} server_type: {}, rtt: {}{}>'.format(self.__class__.__name__, self.address, self.server_type_name, self.round_trip_time, errmsg)
    _host_to_round_trip_time: dict = {}