"""Represent a deployment of MongoDB servers."""
from __future__ import annotations
from random import sample
from typing import Any, Callable, List, Mapping, MutableMapping, NamedTuple, Optional, cast
from bson.min_key import MinKey
from bson.objectid import ObjectId
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref, _ServerMode
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import _Address

class _TopologyType(NamedTuple):
    Single: int
    ReplicaSetNoPrimary: int
    ReplicaSetWithPrimary: int
    Sharded: int
    Unknown: int
    LoadBalanced: int
TOPOLOGY_TYPE = _TopologyType(*range(6))
SRV_POLLING_TOPOLOGIES: tuple[int, int] = (TOPOLOGY_TYPE.Unknown, TOPOLOGY_TYPE.Sharded)
_ServerSelector = Callable[[List[ServerDescription]], List[ServerDescription]]

class TopologyDescription:

    def __init__(self, topology_type: int, server_descriptions: dict[_Address, ServerDescription], replica_set_name: Optional[str], max_set_version: Optional[int], max_election_id: Optional[ObjectId], topology_settings: Any) -> None:
        if False:
            print('Hello World!')
        'Representation of a deployment of MongoDB servers.\n\n        :Parameters:\n          - `topology_type`: initial type\n          - `server_descriptions`: dict of (address, ServerDescription) for\n            all seeds\n          - `replica_set_name`: replica set name or None\n          - `max_set_version`: greatest setVersion seen from a primary, or None\n          - `max_election_id`: greatest electionId seen from a primary, or None\n          - `topology_settings`: a TopologySettings\n        '
        self._topology_type = topology_type
        self._replica_set_name = replica_set_name
        self._server_descriptions = server_descriptions
        self._max_set_version = max_set_version
        self._max_election_id = max_election_id
        self._topology_settings = topology_settings
        self._incompatible_err = None
        if self._topology_type != TOPOLOGY_TYPE.LoadBalanced:
            self._init_incompatible_err()
        readable_servers = self.readable_servers
        if not readable_servers:
            self._ls_timeout_minutes = None
        elif any((s.logical_session_timeout_minutes is None for s in readable_servers)):
            self._ls_timeout_minutes = None
        else:
            self._ls_timeout_minutes = min((s.logical_session_timeout_minutes for s in readable_servers))

    def _init_incompatible_err(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Internal compatibility check for non-load balanced topologies.'
        for s in self._server_descriptions.values():
            if not s.is_server_type_known:
                continue
            server_too_new = s.min_wire_version is not None and s.min_wire_version > common.MAX_SUPPORTED_WIRE_VERSION
            server_too_old = s.max_wire_version is not None and s.max_wire_version < common.MIN_SUPPORTED_WIRE_VERSION
            if server_too_new:
                self._incompatible_err = 'Server at %s:%d requires wire version %d, but this version of PyMongo only supports up to %d.' % (s.address[0], s.address[1] or 0, s.min_wire_version, common.MAX_SUPPORTED_WIRE_VERSION)
            elif server_too_old:
                self._incompatible_err = 'Server at %s:%d reports wire version %d, but this version of PyMongo requires at least %d (MongoDB %s).' % (s.address[0], s.address[1] or 0, s.max_wire_version, common.MIN_SUPPORTED_WIRE_VERSION, common.MIN_SUPPORTED_SERVER_VERSION)
                break

    def check_compatible(self) -> None:
        if False:
            return 10
        "Raise ConfigurationError if any server is incompatible.\n\n        A server is incompatible if its wire protocol version range does not\n        overlap with PyMongo's.\n        "
        if self._incompatible_err:
            raise ConfigurationError(self._incompatible_err)

    def has_server(self, address: _Address) -> bool:
        if False:
            i = 10
            return i + 15
        return address in self._server_descriptions

    def reset_server(self, address: _Address) -> TopologyDescription:
        if False:
            return 10
        'A copy of this description, with one server marked Unknown.'
        unknown_sd = self._server_descriptions[address].to_unknown()
        return updated_topology_description(self, unknown_sd)

    def reset(self) -> TopologyDescription:
        if False:
            for i in range(10):
                print('nop')
        'A copy of this description, with all servers marked Unknown.'
        if self._topology_type == TOPOLOGY_TYPE.ReplicaSetWithPrimary:
            topology_type = TOPOLOGY_TYPE.ReplicaSetNoPrimary
        else:
            topology_type = self._topology_type
        sds = {address: ServerDescription(address) for address in self._server_descriptions}
        return TopologyDescription(topology_type, sds, self._replica_set_name, self._max_set_version, self._max_election_id, self._topology_settings)

    def server_descriptions(self) -> dict[_Address, ServerDescription]:
        if False:
            i = 10
            return i + 15
        'dict of (address,\n        :class:`~pymongo.server_description.ServerDescription`).\n        '
        return self._server_descriptions.copy()

    @property
    def topology_type(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The type of this topology.'
        return self._topology_type

    @property
    def topology_type_name(self) -> str:
        if False:
            while True:
                i = 10
        'The topology type as a human readable string.\n\n        .. versionadded:: 3.4\n        '
        return TOPOLOGY_TYPE._fields[self._topology_type]

    @property
    def replica_set_name(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'The replica set name.'
        return self._replica_set_name

    @property
    def max_set_version(self) -> Optional[int]:
        if False:
            return 10
        'Greatest setVersion seen from a primary, or None.'
        return self._max_set_version

    @property
    def max_election_id(self) -> Optional[ObjectId]:
        if False:
            while True:
                i = 10
        'Greatest electionId seen from a primary, or None.'
        return self._max_election_id

    @property
    def logical_session_timeout_minutes(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        'Minimum logical session timeout, or None.'
        return self._ls_timeout_minutes

    @property
    def known_servers(self) -> list[ServerDescription]:
        if False:
            for i in range(10):
                print('nop')
        'List of Servers of types besides Unknown.'
        return [s for s in self._server_descriptions.values() if s.is_server_type_known]

    @property
    def has_known_servers(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether there are any Servers of types besides Unknown.'
        return any((s for s in self._server_descriptions.values() if s.is_server_type_known))

    @property
    def readable_servers(self) -> list[ServerDescription]:
        if False:
            print('Hello World!')
        'List of readable Servers.'
        return [s for s in self._server_descriptions.values() if s.is_readable]

    @property
    def common_wire_version(self) -> Optional[int]:
        if False:
            print('Hello World!')
        "Minimum of all servers' max wire versions, or None."
        servers = self.known_servers
        if servers:
            return min((s.max_wire_version for s in self.known_servers))
        return None

    @property
    def heartbeat_frequency(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._topology_settings.heartbeat_frequency

    @property
    def srv_max_hosts(self) -> int:
        if False:
            while True:
                i = 10
        return self._topology_settings._srv_max_hosts

    def _apply_local_threshold(self, selection: Optional[Selection]) -> list[ServerDescription]:
        if False:
            print('Hello World!')
        if not selection:
            return []
        fastest = min((cast(float, s.round_trip_time) for s in selection.server_descriptions))
        threshold = self._topology_settings.local_threshold_ms / 1000.0
        return [s for s in selection.server_descriptions if cast(float, s.round_trip_time) - fastest <= threshold]

    def apply_selector(self, selector: Any, address: Optional[_Address]=None, custom_selector: Optional[_ServerSelector]=None) -> list[ServerDescription]:
        if False:
            print('Hello World!')
        'List of servers matching the provided selector(s).\n\n        :Parameters:\n          - `selector`: a callable that takes a Selection as input and returns\n            a Selection as output. For example, an instance of a read\n            preference from :mod:`~pymongo.read_preferences`.\n          - `address` (optional): A server address to select.\n          - `custom_selector` (optional): A callable that augments server\n            selection rules. Accepts a list of\n            :class:`~pymongo.server_description.ServerDescription` objects and\n            return a list of server descriptions that should be considered\n            suitable for the desired operation.\n\n        .. versionadded:: 3.4\n        '
        if getattr(selector, 'min_wire_version', 0):
            common_wv = self.common_wire_version
            if common_wv and common_wv < selector.min_wire_version:
                raise ConfigurationError("%s requires min wire version %d, but topology's min wire version is %d" % (selector, selector.min_wire_version, common_wv))
        if isinstance(selector, _AggWritePref):
            selector.selection_hook(self)
        if self.topology_type == TOPOLOGY_TYPE.Unknown:
            return []
        elif self.topology_type in (TOPOLOGY_TYPE.Single, TOPOLOGY_TYPE.LoadBalanced):
            return self.known_servers
        if address:
            description = self.server_descriptions().get(address)
            return [description] if description else []
        selection = Selection.from_topology_description(self)
        if self.topology_type != TOPOLOGY_TYPE.Sharded:
            selection = selector(selection)
        if custom_selector is not None and selection:
            selection = selection.with_server_descriptions(custom_selector(selection.server_descriptions))
        return self._apply_local_threshold(selection)

    def has_readable_server(self, read_preference: _ServerMode=ReadPreference.PRIMARY) -> bool:
        if False:
            print('Hello World!')
        'Does this topology have any readable servers available matching the\n        given read preference?\n\n        :Parameters:\n          - `read_preference`: an instance of a read preference from\n            :mod:`~pymongo.read_preferences`. Defaults to\n            :attr:`~pymongo.read_preferences.ReadPreference.PRIMARY`.\n\n        .. note:: When connected directly to a single server this method\n          always returns ``True``.\n\n        .. versionadded:: 3.4\n        '
        common.validate_read_preference('read_preference', read_preference)
        return any(self.apply_selector(read_preference))

    def has_writable_server(self) -> bool:
        if False:
            print('Hello World!')
        'Does this topology have a writable server available?\n\n        .. note:: When connected directly to a single server this method\n          always returns ``True``.\n\n        .. versionadded:: 3.4\n        '
        return self.has_readable_server(ReadPreference.PRIMARY)

    def __repr__(self) -> str:
        if False:
            return 10
        servers = sorted(self._server_descriptions.values(), key=lambda sd: sd.address)
        return '<{} id: {}, topology_type: {}, servers: {!r}>'.format(self.__class__.__name__, self._topology_settings._topology_id, self.topology_type_name, servers)
_SERVER_TYPE_TO_TOPOLOGY_TYPE = {SERVER_TYPE.Mongos: TOPOLOGY_TYPE.Sharded, SERVER_TYPE.RSPrimary: TOPOLOGY_TYPE.ReplicaSetWithPrimary, SERVER_TYPE.RSSecondary: TOPOLOGY_TYPE.ReplicaSetNoPrimary, SERVER_TYPE.RSArbiter: TOPOLOGY_TYPE.ReplicaSetNoPrimary, SERVER_TYPE.RSOther: TOPOLOGY_TYPE.ReplicaSetNoPrimary}

def updated_topology_description(topology_description: TopologyDescription, server_description: ServerDescription) -> TopologyDescription:
    if False:
        return 10
    'Return an updated copy of a TopologyDescription.\n\n    :Parameters:\n      - `topology_description`: the current TopologyDescription\n      - `server_description`: a new ServerDescription that resulted from\n        a hello call\n\n    Called after attempting (successfully or not) to call hello on the\n    server at server_description.address. Does not modify topology_description.\n    '
    address = server_description.address
    topology_type = topology_description.topology_type
    set_name = topology_description.replica_set_name
    max_set_version = topology_description.max_set_version
    max_election_id = topology_description.max_election_id
    server_type = server_description.server_type
    sds = topology_description.server_descriptions()
    sds[address] = server_description
    if topology_type == TOPOLOGY_TYPE.Single:
        if set_name is not None and set_name != server_description.replica_set_name:
            error = ConfigurationError("client is configured to connect to a replica set named '{}' but this node belongs to a set named '{}'".format(set_name, server_description.replica_set_name))
            sds[address] = server_description.to_unknown(error=error)
        return TopologyDescription(TOPOLOGY_TYPE.Single, sds, set_name, max_set_version, max_election_id, topology_description._topology_settings)
    if topology_type == TOPOLOGY_TYPE.Unknown:
        if server_type in (SERVER_TYPE.Standalone, SERVER_TYPE.LoadBalancer):
            if len(topology_description._topology_settings.seeds) == 1:
                topology_type = TOPOLOGY_TYPE.Single
            else:
                sds.pop(address)
        elif server_type not in (SERVER_TYPE.Unknown, SERVER_TYPE.RSGhost):
            topology_type = _SERVER_TYPE_TO_TOPOLOGY_TYPE[server_type]
    if topology_type == TOPOLOGY_TYPE.Sharded:
        if server_type not in (SERVER_TYPE.Mongos, SERVER_TYPE.Unknown):
            sds.pop(address)
    elif topology_type == TOPOLOGY_TYPE.ReplicaSetNoPrimary:
        if server_type in (SERVER_TYPE.Standalone, SERVER_TYPE.Mongos):
            sds.pop(address)
        elif server_type == SERVER_TYPE.RSPrimary:
            (topology_type, set_name, max_set_version, max_election_id) = _update_rs_from_primary(sds, set_name, server_description, max_set_version, max_election_id)
        elif server_type in (SERVER_TYPE.RSSecondary, SERVER_TYPE.RSArbiter, SERVER_TYPE.RSOther):
            (topology_type, set_name) = _update_rs_no_primary_from_member(sds, set_name, server_description)
    elif topology_type == TOPOLOGY_TYPE.ReplicaSetWithPrimary:
        if server_type in (SERVER_TYPE.Standalone, SERVER_TYPE.Mongos):
            sds.pop(address)
            topology_type = _check_has_primary(sds)
        elif server_type == SERVER_TYPE.RSPrimary:
            (topology_type, set_name, max_set_version, max_election_id) = _update_rs_from_primary(sds, set_name, server_description, max_set_version, max_election_id)
        elif server_type in (SERVER_TYPE.RSSecondary, SERVER_TYPE.RSArbiter, SERVER_TYPE.RSOther):
            topology_type = _update_rs_with_primary_from_member(sds, set_name, server_description)
        else:
            topology_type = _check_has_primary(sds)
    return TopologyDescription(topology_type, sds, set_name, max_set_version, max_election_id, topology_description._topology_settings)

def _updated_topology_description_srv_polling(topology_description: TopologyDescription, seedlist: list[tuple[str, Any]]) -> TopologyDescription:
    if False:
        print('Hello World!')
    'Return an updated copy of a TopologyDescription.\n\n    :Parameters:\n      - `topology_description`: the current TopologyDescription\n      - `seedlist`: a list of new seeds new ServerDescription that resulted from\n        a hello call\n    '
    assert topology_description.topology_type in SRV_POLLING_TOPOLOGIES
    sds = topology_description.server_descriptions()
    if set(sds.keys()) == set(seedlist):
        return topology_description
    for address in list(sds.keys()):
        if address not in seedlist:
            sds.pop(address)
    if topology_description.srv_max_hosts != 0:
        new_hosts = set(seedlist) - set(sds.keys())
        n_to_add = topology_description.srv_max_hosts - len(sds)
        if n_to_add > 0:
            seedlist = sample(sorted(new_hosts), min(n_to_add, len(new_hosts)))
        else:
            seedlist = []
    for address in seedlist:
        if address not in sds:
            sds[address] = ServerDescription(address)
    return TopologyDescription(topology_description.topology_type, sds, topology_description.replica_set_name, topology_description.max_set_version, topology_description.max_election_id, topology_description._topology_settings)

def _update_rs_from_primary(sds: MutableMapping[_Address, ServerDescription], replica_set_name: Optional[str], server_description: ServerDescription, max_set_version: Optional[int], max_election_id: Optional[ObjectId]) -> tuple[int, Optional[str], Optional[int], Optional[ObjectId]]:
    if False:
        return 10
    "Update topology description from a primary's hello response.\n\n    Pass in a dict of ServerDescriptions, current replica set name, the\n    ServerDescription we are processing, and the TopologyDescription's\n    max_set_version and max_election_id if any.\n\n    Returns (new topology type, new replica_set_name, new max_set_version,\n    new max_election_id).\n    "
    if replica_set_name is None:
        replica_set_name = server_description.replica_set_name
    elif replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
        return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)
    if server_description.max_wire_version is None or server_description.max_wire_version < 17:
        new_election_tuple: tuple = (server_description.set_version, server_description.election_id)
        max_election_tuple: tuple = (max_set_version, max_election_id)
        if None not in new_election_tuple:
            if None not in max_election_tuple and new_election_tuple < max_election_tuple:
                sds[server_description.address] = server_description.to_unknown()
                return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)
            max_election_id = server_description.election_id
        if server_description.set_version is not None and (max_set_version is None or server_description.set_version > max_set_version):
            max_set_version = server_description.set_version
    else:
        new_election_tuple = (server_description.election_id, server_description.set_version)
        max_election_tuple = (max_election_id, max_set_version)
        new_election_safe = tuple((MinKey() if i is None else i for i in new_election_tuple))
        max_election_safe = tuple((MinKey() if i is None else i for i in max_election_tuple))
        if new_election_safe < max_election_safe:
            sds[server_description.address] = server_description.to_unknown()
            return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)
        else:
            max_election_id = server_description.election_id
            max_set_version = server_description.set_version
    for server in sds.values():
        if server.server_type is SERVER_TYPE.RSPrimary and server.address != server_description.address:
            sds[server.address] = server.to_unknown()
            break
    for new_address in server_description.all_hosts:
        if new_address not in sds:
            sds[new_address] = ServerDescription(new_address)
    for addr in set(sds) - server_description.all_hosts:
        sds.pop(addr)
    return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)

def _update_rs_with_primary_from_member(sds: MutableMapping[_Address, ServerDescription], replica_set_name: Optional[str], server_description: ServerDescription) -> int:
    if False:
        print('Hello World!')
    'RS with known primary. Process a response from a non-primary.\n\n    Pass in a dict of ServerDescriptions, current replica set name, and the\n    ServerDescription we are processing.\n\n    Returns new topology type.\n    '
    assert replica_set_name is not None
    if replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
    elif server_description.me and server_description.address != server_description.me:
        sds.pop(server_description.address)
    return _check_has_primary(sds)

def _update_rs_no_primary_from_member(sds: MutableMapping[_Address, ServerDescription], replica_set_name: Optional[str], server_description: ServerDescription) -> tuple[int, Optional[str]]:
    if False:
        i = 10
        return i + 15
    "RS without known primary. Update from a non-primary's response.\n\n    Pass in a dict of ServerDescriptions, current replica set name, and the\n    ServerDescription we are processing.\n\n    Returns (new topology type, new replica_set_name).\n    "
    topology_type = TOPOLOGY_TYPE.ReplicaSetNoPrimary
    if replica_set_name is None:
        replica_set_name = server_description.replica_set_name
    elif replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
        return (topology_type, replica_set_name)
    for address in server_description.all_hosts:
        if address not in sds:
            sds[address] = ServerDescription(address)
    if server_description.me and server_description.address != server_description.me:
        sds.pop(server_description.address)
    return (topology_type, replica_set_name)

def _check_has_primary(sds: Mapping[_Address, ServerDescription]) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Current topology type is ReplicaSetWithPrimary. Is primary still known?\n\n    Pass in a dict of ServerDescriptions.\n\n    Returns new topology type.\n    '
    for s in sds.values():
        if s.server_type == SERVER_TYPE.RSPrimary:
            return TOPOLOGY_TYPE.ReplicaSetWithPrimary
    else:
        return TOPOLOGY_TYPE.ReplicaSetNoPrimary