"""Utilities for choosing which member of a replica set to read from."""
from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import member_with_tags_server_selector, secondary_with_tags_server_selector
if TYPE_CHECKING:
    from pymongo.server_selectors import Selection
    from pymongo.topology_description import TopologyDescription
_PRIMARY = 0
_PRIMARY_PREFERRED = 1
_SECONDARY = 2
_SECONDARY_PREFERRED = 3
_NEAREST = 4
_MONGOS_MODES = ('primary', 'primaryPreferred', 'secondary', 'secondaryPreferred', 'nearest')
_Hedge = Mapping[str, Any]
_TagSets = Sequence[Mapping[str, Any]]

def _validate_tag_sets(tag_sets: Optional[_TagSets]) -> Optional[_TagSets]:
    if False:
        return 10
    'Validate tag sets for a MongoClient.'
    if tag_sets is None:
        return tag_sets
    if not isinstance(tag_sets, (list, tuple)):
        raise TypeError(f'Tag sets {tag_sets!r} invalid, must be a sequence')
    if len(tag_sets) == 0:
        raise ValueError(f'Tag sets {tag_sets!r} invalid, must be None or contain at least one set of tags')
    for tags in tag_sets:
        if not isinstance(tags, abc.Mapping):
            raise TypeError(f'Tag set {tags!r} invalid, must be an instance of dict, bson.son.SON or other type that inherits from collection.Mapping')
    return list(tag_sets)

def _invalid_max_staleness_msg(max_staleness: Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'maxStalenessSeconds must be a positive integer, not %s' % max_staleness

def _validate_max_staleness(max_staleness: Any) -> int:
    if False:
        i = 10
        return i + 15
    'Validate max_staleness.'
    if max_staleness == -1:
        return -1
    if not isinstance(max_staleness, int):
        raise TypeError(_invalid_max_staleness_msg(max_staleness))
    if max_staleness <= 0:
        raise ValueError(_invalid_max_staleness_msg(max_staleness))
    return max_staleness

def _validate_hedge(hedge: Optional[_Hedge]) -> Optional[_Hedge]:
    if False:
        while True:
            i = 10
    'Validate hedge.'
    if hedge is None:
        return None
    if not isinstance(hedge, dict):
        raise TypeError(f'hedge must be a dictionary, not {hedge!r}')
    return hedge

class _ServerMode:
    """Base class for all read preferences."""
    __slots__ = ('__mongos_mode', '__mode', '__tag_sets', '__max_staleness', '__hedge')

    def __init__(self, mode: int, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        if False:
            print('Hello World!')
        self.__mongos_mode = _MONGOS_MODES[mode]
        self.__mode = mode
        self.__tag_sets = _validate_tag_sets(tag_sets)
        self.__max_staleness = _validate_max_staleness(max_staleness)
        self.__hedge = _validate_hedge(hedge)

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        'The name of this read preference.'
        return self.__class__.__name__

    @property
    def mongos_mode(self) -> str:
        if False:
            print('Hello World!')
        'The mongos mode of this read preference.'
        return self.__mongos_mode

    @property
    def document(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Read preference as a document.'
        doc: dict[str, Any] = {'mode': self.__mongos_mode}
        if self.__tag_sets not in (None, [{}]):
            doc['tags'] = self.__tag_sets
        if self.__max_staleness != -1:
            doc['maxStalenessSeconds'] = self.__max_staleness
        if self.__hedge not in (None, {}):
            doc['hedge'] = self.__hedge
        return doc

    @property
    def mode(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The mode of this read preference instance.'
        return self.__mode

    @property
    def tag_sets(self) -> _TagSets:
        if False:
            print('Hello World!')
        'Set ``tag_sets`` to a list of dictionaries like [{\'dc\': \'ny\'}] to\n        read only from members whose ``dc`` tag has the value ``"ny"``.\n        To specify a priority-order for tag sets, provide a list of\n        tag sets: ``[{\'dc\': \'ny\'}, {\'dc\': \'la\'}, {}]``. A final, empty tag\n        set, ``{}``, means "read from any member that matches the mode,\n        ignoring tags." MongoClient tries each set of tags in turn\n        until it finds a set of tags with at least one matching member.\n        For example, to only send a query to an analytic node::\n\n           Nearest(tag_sets=[{"node":"analytics"}])\n\n        Or using :class:`SecondaryPreferred`::\n\n           SecondaryPreferred(tag_sets=[{"node":"analytics"}])\n\n           .. seealso:: `Data-Center Awareness\n               <https://www.mongodb.com/docs/manual/data-center-awareness/>`_\n        '
        return list(self.__tag_sets) if self.__tag_sets else [{}]

    @property
    def max_staleness(self) -> int:
        if False:
            print('Hello World!')
        'The maximum estimated length of time (in seconds) a replica set\n        secondary can fall behind the primary in replication before it will\n        no longer be selected for operations, or -1 for no maximum.\n        '
        return self.__max_staleness

    @property
    def hedge(self) -> Optional[_Hedge]:
        if False:
            print('Hello World!')
        "The read preference ``hedge`` parameter.\n\n        A dictionary that configures how the server will perform hedged reads.\n        It consists of the following keys:\n\n        - ``enabled``: Enables or disables hedged reads in sharded clusters.\n\n        Hedged reads are automatically enabled in MongoDB 4.4+ when using a\n        ``nearest`` read preference. To explicitly enable hedged reads, set\n        the ``enabled`` key  to ``true``::\n\n            >>> Nearest(hedge={'enabled': True})\n\n        To explicitly disable hedged reads, set the ``enabled`` key  to\n        ``False``::\n\n            >>> Nearest(hedge={'enabled': False})\n\n        .. versionadded:: 3.11\n        "
        return self.__hedge

    @property
    def min_wire_version(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "The wire protocol version the server must support.\n\n        Some read preferences impose version requirements on all servers (e.g.\n        maxStalenessSeconds requires MongoDB 3.4 / maxWireVersion 5).\n\n        All servers' maxWireVersion must be at least this read preference's\n        `min_wire_version`, or the driver raises\n        :exc:`~pymongo.errors.ConfigurationError`.\n        "
        return 0 if self.__max_staleness == -1 else 5

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '{}(tag_sets={!r}, max_staleness={!r}, hedge={!r})'.format(self.name, self.__tag_sets, self.__max_staleness, self.__hedge)

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, _ServerMode):
            return self.mode == other.mode and self.tag_sets == other.tag_sets and (self.max_staleness == other.max_staleness) and (self.hedge == other.hedge)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __getstate__(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return value of object for pickling.\n\n        Needed explicitly because __slots__() defined.\n        '
        return {'mode': self.__mode, 'tag_sets': self.__tag_sets, 'max_staleness': self.__max_staleness, 'hedge': self.__hedge}

    def __setstate__(self, value: Mapping[str, Any]) -> None:
        if False:
            while True:
                i = 10
        'Restore from pickling.'
        self.__mode = value['mode']
        self.__mongos_mode = _MONGOS_MODES[self.__mode]
        self.__tag_sets = _validate_tag_sets(value['tag_sets'])
        self.__max_staleness = _validate_max_staleness(value['max_staleness'])
        self.__hedge = _validate_hedge(value['hedge'])

    def __call__(self, selection: Selection) -> Selection:
        if False:
            return 10
        return selection

class Primary(_ServerMode):
    """Primary read preference.

    * When directly connected to one mongod queries are allowed if the server
      is standalone or a replica set primary.
    * When connected to a mongos queries are sent to the primary of a shard.
    * When connected to a replica set queries are sent to the primary of
      the replica set.
    """
    __slots__ = ()

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__(_PRIMARY)

    def __call__(self, selection: Selection) -> Selection:
        if False:
            while True:
                i = 10
        'Apply this read preference to a Selection.'
        return selection.primary_selection

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Primary()'

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, _ServerMode):
            return other.mode == _PRIMARY
        return NotImplemented

class PrimaryPreferred(_ServerMode):
    """PrimaryPreferred read preference.

    * When directly connected to one mongod queries are allowed to standalone
      servers, to a replica set primary, or to replica set secondaries.
    * When connected to a mongos queries are sent to the primary of a shard if
      available, otherwise a shard secondary.
    * When connected to a replica set queries are sent to the primary if
      available, otherwise a secondary.

    .. note:: When a :class:`~pymongo.mongo_client.MongoClient` is first
      created reads will be routed to an available secondary until the
      primary of the replica set is discovered.

    :Parameters:
      - `tag_sets`: The :attr:`~tag_sets` to use if the primary is not
        available.
      - `max_staleness`: (integer, in seconds) The maximum estimated
        length of time a replica set secondary can fall behind the primary in
        replication before it will no longer be selected for operations.
        Default -1, meaning no maximum. If it is set, it must be at least
        90 seconds.
      - `hedge`: The :attr:`~hedge` to use if the primary is not available.

    .. versionchanged:: 3.11
       Added ``hedge`` parameter.
    """
    __slots__ = ()

    def __init__(self, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(_PRIMARY_PREFERRED, tag_sets, max_staleness, hedge)

    def __call__(self, selection: Selection) -> Selection:
        if False:
            for i in range(10):
                print('nop')
        'Apply this read preference to Selection.'
        if selection.primary:
            return selection.primary_selection
        else:
            return secondary_with_tags_server_selector(self.tag_sets, max_staleness_selectors.select(self.max_staleness, selection))

class Secondary(_ServerMode):
    """Secondary read preference.

    * When directly connected to one mongod queries are allowed to standalone
      servers, to a replica set primary, or to replica set secondaries.
    * When connected to a mongos queries are distributed among shard
      secondaries. An error is raised if no secondaries are available.
    * When connected to a replica set queries are distributed among
      secondaries. An error is raised if no secondaries are available.

    :Parameters:
      - `tag_sets`: The :attr:`~tag_sets` for this read preference.
      - `max_staleness`: (integer, in seconds) The maximum estimated
        length of time a replica set secondary can fall behind the primary in
        replication before it will no longer be selected for operations.
        Default -1, meaning no maximum. If it is set, it must be at least
        90 seconds.
      - `hedge`: The :attr:`~hedge` for this read preference.

    .. versionchanged:: 3.11
       Added ``hedge`` parameter.
    """
    __slots__ = ()

    def __init__(self, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(_SECONDARY, tag_sets, max_staleness, hedge)

    def __call__(self, selection: Selection) -> Selection:
        if False:
            print('Hello World!')
        'Apply this read preference to Selection.'
        return secondary_with_tags_server_selector(self.tag_sets, max_staleness_selectors.select(self.max_staleness, selection))

class SecondaryPreferred(_ServerMode):
    """SecondaryPreferred read preference.

    * When directly connected to one mongod queries are allowed to standalone
      servers, to a replica set primary, or to replica set secondaries.
    * When connected to a mongos queries are distributed among shard
      secondaries, or the shard primary if no secondary is available.
    * When connected to a replica set queries are distributed among
      secondaries, or the primary if no secondary is available.

    .. note:: When a :class:`~pymongo.mongo_client.MongoClient` is first
      created reads will be routed to the primary of the replica set until
      an available secondary is discovered.

    :Parameters:
      - `tag_sets`: The :attr:`~tag_sets` for this read preference.
      - `max_staleness`: (integer, in seconds) The maximum estimated
        length of time a replica set secondary can fall behind the primary in
        replication before it will no longer be selected for operations.
        Default -1, meaning no maximum. If it is set, it must be at least
        90 seconds.
      - `hedge`: The :attr:`~hedge` for this read preference.

    .. versionchanged:: 3.11
       Added ``hedge`` parameter.
    """
    __slots__ = ()

    def __init__(self, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(_SECONDARY_PREFERRED, tag_sets, max_staleness, hedge)

    def __call__(self, selection: Selection) -> Selection:
        if False:
            print('Hello World!')
        'Apply this read preference to Selection.'
        secondaries = secondary_with_tags_server_selector(self.tag_sets, max_staleness_selectors.select(self.max_staleness, selection))
        if secondaries:
            return secondaries
        else:
            return selection.primary_selection

class Nearest(_ServerMode):
    """Nearest read preference.

    * When directly connected to one mongod queries are allowed to standalone
      servers, to a replica set primary, or to replica set secondaries.
    * When connected to a mongos queries are distributed among all members of
      a shard.
    * When connected to a replica set queries are distributed among all
      members.

    :Parameters:
      - `tag_sets`: The :attr:`~tag_sets` for this read preference.
      - `max_staleness`: (integer, in seconds) The maximum estimated
        length of time a replica set secondary can fall behind the primary in
        replication before it will no longer be selected for operations.
        Default -1, meaning no maximum. If it is set, it must be at least
        90 seconds.
      - `hedge`: The :attr:`~hedge` for this read preference.

    .. versionchanged:: 3.11
       Added ``hedge`` parameter.
    """
    __slots__ = ()

    def __init__(self, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(_NEAREST, tag_sets, max_staleness, hedge)

    def __call__(self, selection: Selection) -> Selection:
        if False:
            while True:
                i = 10
        'Apply this read preference to Selection.'
        return member_with_tags_server_selector(self.tag_sets, max_staleness_selectors.select(self.max_staleness, selection))

class _AggWritePref:
    """Agg $out/$merge write preference.

    * If there are readable servers and there is any pre-5.0 server, use
      primary read preference.
    * Otherwise use `pref` read preference.

    :Parameters:
      - `pref`: The read preference to use on MongoDB 5.0+.
    """
    __slots__ = ('pref', 'effective_pref')

    def __init__(self, pref: _ServerMode):
        if False:
            print('Hello World!')
        self.pref = pref
        self.effective_pref: _ServerMode = ReadPreference.PRIMARY

    def selection_hook(self, topology_description: TopologyDescription) -> None:
        if False:
            for i in range(10):
                print('nop')
        common_wv = topology_description.common_wire_version
        if topology_description.has_readable_server(ReadPreference.PRIMARY_PREFERRED) and common_wv and (common_wv < 13):
            self.effective_pref = ReadPreference.PRIMARY
        else:
            self.effective_pref = self.pref

    def __call__(self, selection: Selection) -> Selection:
        if False:
            print('Hello World!')
        'Apply this read preference to a Selection.'
        return self.effective_pref(selection)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'_AggWritePref(pref={self.pref!r})'

    def __getattr__(self, name: str) -> Any:
        if False:
            i = 10
            return i + 15
        return getattr(self.effective_pref, name)
_ALL_READ_PREFERENCES = (Primary, PrimaryPreferred, Secondary, SecondaryPreferred, Nearest)

def make_read_preference(mode: int, tag_sets: Optional[_TagSets], max_staleness: int=-1) -> _ServerMode:
    if False:
        print('Hello World!')
    if mode == _PRIMARY:
        if tag_sets not in (None, [{}]):
            raise ConfigurationError('Read preference primary cannot be combined with tags')
        if max_staleness != -1:
            raise ConfigurationError('Read preference primary cannot be combined with maxStalenessSeconds')
        return Primary()
    return _ALL_READ_PREFERENCES[mode](tag_sets, max_staleness)
_MODES = ('PRIMARY', 'PRIMARY_PREFERRED', 'SECONDARY', 'SECONDARY_PREFERRED', 'NEAREST')

class ReadPreference:
    """An enum that defines some commonly used read preference modes.

    Apps can also create a custom read preference, for example::

       Nearest(tag_sets=[{"node":"analytics"}])

    See :doc:`/examples/high_availability` for code examples.

    A read preference is used in three cases:

    :class:`~pymongo.mongo_client.MongoClient` connected to a single mongod:

    - ``PRIMARY``: Queries are allowed if the server is standalone or a replica
      set primary.
    - All other modes allow queries to standalone servers, to a replica set
      primary, or to replica set secondaries.

    :class:`~pymongo.mongo_client.MongoClient` initialized with the
    ``replicaSet`` option:

    - ``PRIMARY``: Read from the primary. This is the default, and provides the
      strongest consistency. If no primary is available, raise
      :class:`~pymongo.errors.AutoReconnect`.

    - ``PRIMARY_PREFERRED``: Read from the primary if available, or if there is
      none, read from a secondary.

    - ``SECONDARY``: Read from a secondary. If no secondary is available,
      raise :class:`~pymongo.errors.AutoReconnect`.

    - ``SECONDARY_PREFERRED``: Read from a secondary if available, otherwise
      from the primary.

    - ``NEAREST``: Read from any member.

    :class:`~pymongo.mongo_client.MongoClient` connected to a mongos, with a
    sharded cluster of replica sets:

    - ``PRIMARY``: Read from the primary of the shard, or raise
      :class:`~pymongo.errors.OperationFailure` if there is none.
      This is the default.

    - ``PRIMARY_PREFERRED``: Read from the primary of the shard, or if there is
      none, read from a secondary of the shard.

    - ``SECONDARY``: Read from a secondary of the shard, or raise
      :class:`~pymongo.errors.OperationFailure` if there is none.

    - ``SECONDARY_PREFERRED``: Read from a secondary of the shard if available,
      otherwise from the shard primary.

    - ``NEAREST``: Read from any shard member.
    """
    PRIMARY = Primary()
    PRIMARY_PREFERRED = PrimaryPreferred()
    SECONDARY = Secondary()
    SECONDARY_PREFERRED = SecondaryPreferred()
    NEAREST = Nearest()

def read_pref_mode_from_name(name: str) -> int:
    if False:
        return 10
    'Get the read preference mode from mongos/uri name.'
    return _MONGOS_MODES.index(name)

class MovingAverage:
    """Tracks an exponentially-weighted moving average."""
    average: Optional[float]

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.average = None

    def add_sample(self, sample: float) -> None:
        if False:
            print('Hello World!')
        if sample < 0:
            return
        if self.average is None:
            self.average = sample
        else:
            self.average = 0.8 * self.average + 0.2 * sample

    def get(self) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        'Get the calculated average, or None if no samples yet.'
        return self.average

    def reset(self) -> None:
        if False:
            print('Hello World!')
        self.average = None