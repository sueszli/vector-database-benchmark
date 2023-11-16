"""Models for SQLAlchemy.

This file contains the model definitions for schema version 30.
It is used to test the schema migration logic.
"""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
import time
from typing import Any, Self, TypedDict, cast, overload
import ciso8601
from fnv_hash_fast import fnv1a_32
from sqlalchemy import JSON, BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer, LargeBinary, SmallInteger, String, Text, distinct, type_coerce
from sqlalchemy.dialects import mysql, oracle, postgresql, sqlite
from sqlalchemy.orm import aliased, declarative_base, relationship
from sqlalchemy.orm.session import Session
from homeassistant.components.recorder.const import SupportedDialect
from homeassistant.const import ATTR_ATTRIBUTION, ATTR_RESTORED, ATTR_SUPPORTED_FEATURES, MAX_LENGTH_EVENT_CONTEXT_ID, MAX_LENGTH_EVENT_EVENT_TYPE, MAX_LENGTH_EVENT_ORIGIN, MAX_LENGTH_STATE_ENTITY_ID, MAX_LENGTH_STATE_STATE
from homeassistant.core import Context, Event, EventOrigin, State, split_entity_id
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.json import JSON_DUMP, json_bytes
import homeassistant.util.dt as dt_util
from homeassistant.util.json import JSON_DECODE_EXCEPTIONS, json_loads
ALL_DOMAIN_EXCLUDE_ATTRS = {ATTR_ATTRIBUTION, ATTR_RESTORED, ATTR_SUPPORTED_FEATURES}
Base = declarative_base()
SCHEMA_VERSION = 30
_LOGGER = logging.getLogger(__name__)
TABLE_EVENTS = 'events'
TABLE_EVENT_DATA = 'event_data'
TABLE_EVENT_TYPES = 'event_types'
TABLE_STATES = 'states'
TABLE_STATE_ATTRIBUTES = 'state_attributes'
TABLE_STATES_META = 'states_meta'
TABLE_RECORDER_RUNS = 'recorder_runs'
TABLE_SCHEMA_CHANGES = 'schema_changes'
TABLE_STATISTICS = 'statistics'
TABLE_STATISTICS_META = 'statistics_meta'
TABLE_STATISTICS_RUNS = 'statistics_runs'
TABLE_STATISTICS_SHORT_TERM = 'statistics_short_term'
ALL_TABLES = [TABLE_STATES, TABLE_STATE_ATTRIBUTES, TABLE_STATES_META, TABLE_EVENTS, TABLE_EVENT_DATA, TABLE_EVENT_TYPES, TABLE_RECORDER_RUNS, TABLE_SCHEMA_CHANGES, TABLE_STATISTICS, TABLE_STATISTICS_META, TABLE_STATISTICS_RUNS, TABLE_STATISTICS_SHORT_TERM]
TABLES_TO_CHECK = [TABLE_STATES, TABLE_EVENTS, TABLE_RECORDER_RUNS, TABLE_SCHEMA_CHANGES]
LAST_UPDATED_INDEX = 'ix_states_last_updated'
ENTITY_ID_LAST_UPDATED_INDEX = 'ix_states_entity_id_last_updated'
EVENTS_CONTEXT_ID_INDEX = 'ix_events_context_id'
STATES_CONTEXT_ID_INDEX = 'ix_states_context_id'
CONTEXT_ID_BIN_MAX_LENGTH = 16
EVENTS_CONTEXT_ID_BIN_INDEX = 'ix_events_context_id_bin'
STATES_CONTEXT_ID_BIN_INDEX = 'ix_states_context_id_bin'

class FAST_PYSQLITE_DATETIME(sqlite.DATETIME):
    """Use ciso8601 to parse datetimes instead of sqlalchemy built-in regex."""

    def result_processor(self, dialect, coltype):
        if False:
            while True:
                i = 10
        'Offload the datetime parsing to ciso8601.'
        return lambda value: None if value is None else ciso8601.parse_datetime(value)
JSON_VARIANT_CAST = Text().with_variant(postgresql.JSON(none_as_null=True), 'postgresql')
JSONB_VARIANT_CAST = Text().with_variant(postgresql.JSONB(none_as_null=True), 'postgresql')
DATETIME_TYPE = DateTime(timezone=True).with_variant(mysql.DATETIME(timezone=True, fsp=6), 'mysql').with_variant(FAST_PYSQLITE_DATETIME(), 'sqlite')
DOUBLE_TYPE = Float().with_variant(mysql.DOUBLE(asdecimal=False), 'mysql').with_variant(oracle.DOUBLE_PRECISION(), 'oracle').with_variant(postgresql.DOUBLE_PRECISION(), 'postgresql')
TIMESTAMP_TYPE = DOUBLE_TYPE

class UnsupportedDialect(Exception):
    """The dialect or its version is not supported."""

class StatisticResult(TypedDict):
    """Statistic result data class.

    Allows multiple datapoints for the same statistic_id.
    """
    meta: StatisticMetaData
    stat: StatisticData

class StatisticDataBase(TypedDict):
    """Mandatory fields for statistic data class."""
    start: datetime

class StatisticData(StatisticDataBase, total=False):
    """Statistic data class."""
    mean: float
    min: float
    max: float
    last_reset: datetime | None
    state: float
    sum: float

class StatisticMetaData(TypedDict):
    """Statistic meta data class."""
    has_mean: bool
    has_sum: bool
    name: str | None
    source: str
    statistic_id: str
    unit_of_measurement: str | None

class JSONLiteral(JSON):
    """Teach SA how to literalize json."""

    def literal_processor(self, dialect: str) -> Callable[[Any], str]:
        if False:
            return 10
        'Processor to convert a value to JSON.'

        def process(value: Any) -> str:
            if False:
                return 10
            'Dump json.'
            return JSON_DUMP(value)
        return process
EVENT_ORIGIN_ORDER = [EventOrigin.local, EventOrigin.remote]
EVENT_ORIGIN_TO_IDX = {origin: idx for (idx, origin) in enumerate(EVENT_ORIGIN_ORDER)}

class Events(Base):
    """Event history data."""
    __table_args__ = (Index('ix_events_event_type_time_fired', 'event_type', 'time_fired'), Index(EVENTS_CONTEXT_ID_BIN_INDEX, 'context_id_bin', mysql_length=CONTEXT_ID_BIN_MAX_LENGTH, mariadb_length=CONTEXT_ID_BIN_MAX_LENGTH), {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'})
    __tablename__ = TABLE_EVENTS
    event_id = Column(Integer, Identity(), primary_key=True)
    event_type = Column(String(MAX_LENGTH_EVENT_EVENT_TYPE))
    event_data = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    origin = Column(String(MAX_LENGTH_EVENT_ORIGIN))
    origin_idx = Column(SmallInteger)
    time_fired = Column(DATETIME_TYPE, index=True)
    time_fired_ts = Column(TIMESTAMP_TYPE, index=True)
    context_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_user_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID))
    context_parent_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID))
    data_id = Column(Integer, ForeignKey('event_data.data_id'), index=True)
    context_id_bin = Column(LargeBinary(CONTEXT_ID_BIN_MAX_LENGTH))
    context_user_id_bin = Column(LargeBinary(CONTEXT_ID_BIN_MAX_LENGTH))
    context_parent_id_bin = Column(LargeBinary(CONTEXT_ID_BIN_MAX_LENGTH))
    event_type_id = Column(Integer, ForeignKey('event_types.event_type_id'), index=True)
    event_data_rel = relationship('EventData')
    event_type_rel = relationship('EventTypes')

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        'Return string representation of instance for debugging.'
        return f"<recorder.Events(id={self.event_id}, type='{self.event_type}', origin_idx='{self.origin_idx}', time_fired='{self.time_fired}', data_id={self.data_id})>"

    @staticmethod
    def from_event(event: Event) -> Events:
        if False:
            print('Hello World!')
        'Create an event database object from a native event.'
        return Events(event_type=event.event_type, event_data=None, origin_idx=EVENT_ORIGIN_TO_IDX.get(event.origin), time_fired=event.time_fired, context_id=event.context.id, context_user_id=event.context.user_id, context_parent_id=event.context.parent_id)

    def to_native(self, validate_entity_id: bool=True) -> Event | None:
        if False:
            for i in range(10):
                print('nop')
        'Convert to a native HA Event.'
        context = Context(id=self.context_id, user_id=self.context_user_id, parent_id=self.context_parent_id)
        try:
            return Event(self.event_type, json_loads(self.event_data) if self.event_data else {}, EventOrigin(self.origin) if self.origin else EVENT_ORIGIN_ORDER[self.origin_idx], process_timestamp(self.time_fired), context=context)
        except JSON_DECODE_EXCEPTIONS:
            _LOGGER.exception('Error converting to event: %s', self)
            return None

class EventData(Base):
    """Event data history."""
    __table_args__ = ({'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},)
    __tablename__ = TABLE_EVENT_DATA
    data_id = Column(Integer, Identity(), primary_key=True)
    hash = Column(BigInteger, index=True)
    shared_data = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))

    def __repr__(self) -> str:
        if False:
            return 10
        'Return string representation of instance for debugging.'
        return f"<recorder.EventData(id={self.data_id}, hash='{self.hash}', data='{self.shared_data}')>"

    @staticmethod
    def from_event(event: Event) -> EventData:
        if False:
            while True:
                i = 10
        'Create object from an event.'
        shared_data = json_bytes(event.data)
        return EventData(shared_data=shared_data.decode('utf-8'), hash=EventData.hash_shared_data_bytes(shared_data))

    @staticmethod
    def shared_data_bytes_from_event(event: Event, dialect: SupportedDialect | None) -> bytes:
        if False:
            return 10
        'Create shared_data from an event.'
        return json_bytes(event.data)

    @staticmethod
    def hash_shared_data_bytes(shared_data_bytes: bytes) -> int:
        if False:
            print('Hello World!')
        'Return the hash of json encoded shared data.'
        return cast(int, fnv1a_32(shared_data_bytes))

    def to_native(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Convert to an HA state object.'
        try:
            return cast(dict[str, Any], json_loads(self.shared_data))
        except JSON_DECODE_EXCEPTIONS:
            _LOGGER.exception('Error converting row to event data: %s', self)
            return {}

class EventTypes(Base):
    """Event type history."""
    __table_args__ = ({'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},)
    __tablename__ = TABLE_EVENT_TYPES
    event_type_id = Column(Integer, Identity(), primary_key=True)
    event_type = Column(String(MAX_LENGTH_EVENT_EVENT_TYPE))

class States(Base):
    """State change history."""
    __table_args__ = (Index(ENTITY_ID_LAST_UPDATED_INDEX, 'entity_id', 'last_updated'), Index(STATES_CONTEXT_ID_BIN_INDEX, 'context_id_bin', mysql_length=CONTEXT_ID_BIN_MAX_LENGTH, mariadb_length=CONTEXT_ID_BIN_MAX_LENGTH), {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'})
    __tablename__ = TABLE_STATES
    state_id = Column(Integer, Identity(), primary_key=True)
    entity_id = Column(String(MAX_LENGTH_STATE_ENTITY_ID))
    state = Column(String(MAX_LENGTH_STATE_STATE))
    attributes = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    event_id = Column(Integer, ForeignKey('events.event_id', ondelete='CASCADE'), index=True)
    last_changed = Column(DATETIME_TYPE)
    last_changed_ts = Column(TIMESTAMP_TYPE)
    last_updated = Column(DATETIME_TYPE, default=dt_util.utcnow, index=True)
    last_updated_ts = Column(TIMESTAMP_TYPE, default=time.time, index=True)
    old_state_id = Column(Integer, ForeignKey('states.state_id'), index=True)
    attributes_id = Column(Integer, ForeignKey('state_attributes.attributes_id'), index=True)
    context_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_user_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID))
    context_parent_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID))
    origin_idx = Column(SmallInteger)
    context_id_bin = Column(LargeBinary(CONTEXT_ID_BIN_MAX_LENGTH))
    context_user_id_bin = Column(LargeBinary(CONTEXT_ID_BIN_MAX_LENGTH))
    context_parent_id_bin = Column(LargeBinary(CONTEXT_ID_BIN_MAX_LENGTH))
    metadata_id = Column(Integer, ForeignKey('states_meta.metadata_id'), index=True)
    states_meta_rel = relationship('StatesMeta')
    old_state = relationship('States', remote_side=[state_id])
    state_attributes = relationship('StateAttributes')

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        'Return string representation of instance for debugging.'
        return f"<recorder.States(id={self.state_id}, entity_id='{self.entity_id}', state='{self.state}', event_id='{self.event_id}', last_updated='{self.last_updated.isoformat(sep=' ', timespec='seconds')}', old_state_id={self.old_state_id}, attributes_id={self.attributes_id})>"

    @staticmethod
    def from_event(event: Event) -> States:
        if False:
            for i in range(10):
                print('nop')
        'Create object from a state_changed event.'
        entity_id = event.data['entity_id']
        state: State | None = event.data.get('new_state')
        dbstate = States(entity_id=entity_id, attributes=None, context_id=event.context.id, context_user_id=event.context.user_id, context_parent_id=event.context.parent_id, origin_idx=EVENT_ORIGIN_TO_IDX.get(event.origin))
        if state is None:
            dbstate.state = ''
            dbstate.last_updated = event.time_fired
            dbstate.last_changed = None
            return dbstate
        dbstate.state = state.state
        dbstate.last_updated = state.last_updated
        if state.last_updated == state.last_changed:
            dbstate.last_changed = None
        else:
            dbstate.last_changed = state.last_changed
        return dbstate

    def to_native(self, validate_entity_id: bool=True) -> State | None:
        if False:
            print('Hello World!')
        'Convert to an HA state object.'
        context = Context(id=self.context_id, user_id=self.context_user_id, parent_id=self.context_parent_id)
        try:
            attrs = json_loads(self.attributes) if self.attributes else {}
        except JSON_DECODE_EXCEPTIONS:
            _LOGGER.exception('Error converting row to state: %s', self)
            return None
        if self.last_changed is None or self.last_changed == self.last_updated:
            last_changed = last_updated = process_timestamp(self.last_updated)
        else:
            last_updated = process_timestamp(self.last_updated)
            last_changed = process_timestamp(self.last_changed)
        return State(self.entity_id, self.state, attrs, last_changed, last_updated, context=context, validate_entity_id=validate_entity_id)

class StateAttributes(Base):
    """State attribute change history."""
    __table_args__ = ({'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},)
    __tablename__ = TABLE_STATE_ATTRIBUTES
    attributes_id = Column(Integer, Identity(), primary_key=True)
    hash = Column(BigInteger, index=True)
    shared_attrs = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return string representation of instance for debugging.'
        return f"<recorder.StateAttributes(id={self.attributes_id}, hash='{self.hash}', attributes='{self.shared_attrs}')>"

    @staticmethod
    def from_event(event: Event) -> StateAttributes:
        if False:
            i = 10
            return i + 15
        'Create object from a state_changed event.'
        state: State | None = event.data.get('new_state')
        attr_bytes = b'{}' if state is None else json_bytes(state.attributes)
        dbstate = StateAttributes(shared_attrs=attr_bytes.decode('utf-8'))
        dbstate.hash = StateAttributes.hash_shared_attrs_bytes(attr_bytes)
        return dbstate

    @staticmethod
    def shared_attrs_bytes_from_event(event: Event, entity_registry: er.EntityRegistry, exclude_attrs_by_domain: dict[str, set[str]], dialect: SupportedDialect | None) -> bytes:
        if False:
            while True:
                i = 10
        'Create shared_attrs from a state_changed event.'
        state: State | None = event.data.get('new_state')
        if state is None:
            return b'{}'
        domain = split_entity_id(state.entity_id)[0]
        exclude_attrs = exclude_attrs_by_domain.get(domain, set()) | ALL_DOMAIN_EXCLUDE_ATTRS
        return json_bytes({k: v for (k, v) in state.attributes.items() if k not in exclude_attrs})

    @staticmethod
    def hash_shared_attrs_bytes(shared_attrs_bytes: bytes) -> int:
        if False:
            print('Hello World!')
        'Return the hash of json encoded shared attributes.'
        return cast(int, fnv1a_32(shared_attrs_bytes))

    def to_native(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Convert to an HA state object.'
        try:
            return cast(dict[str, Any], json_loads(self.shared_attrs))
        except JSON_DECODE_EXCEPTIONS:
            _LOGGER.exception('Error converting row to state attributes: %s', self)
            return {}

class StatesMeta(Base):
    """Metadata for states."""
    __table_args__ = ({'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},)
    __tablename__ = TABLE_STATES_META
    metadata_id = Column(Integer, Identity(), primary_key=True)
    entity_id = Column(String(MAX_LENGTH_STATE_ENTITY_ID))

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return string representation of instance for debugging.'
        return f"<recorder.StatesMeta(id={self.metadata_id}, entity_id='{self.entity_id}')>"

class StatisticsBase:
    """Statistics base class."""
    id = Column(Integer, Identity(), primary_key=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    metadata_id = Column(Integer, ForeignKey(f'{TABLE_STATISTICS_META}.id', ondelete='CASCADE'), index=True)
    start = Column(DATETIME_TYPE, index=True)
    mean = Column(DOUBLE_TYPE)
    min = Column(DOUBLE_TYPE)
    max = Column(DOUBLE_TYPE)
    last_reset = Column(DATETIME_TYPE)
    state = Column(DOUBLE_TYPE)
    sum = Column(DOUBLE_TYPE)

    @classmethod
    def from_stats(cls, metadata_id: int, stats: StatisticData) -> Self:
        if False:
            i = 10
            return i + 15
        'Create object from a statistics.'
        return cls(metadata_id=metadata_id, **stats)

class Statistics(Base, StatisticsBase):
    """Long term statistics."""
    duration = timedelta(hours=1)
    __table_args__ = (Index('ix_statistics_statistic_id_start', 'metadata_id', 'start', unique=True),)
    __tablename__ = TABLE_STATISTICS

class StatisticsShortTerm(Base, StatisticsBase):
    """Short term statistics."""
    duration = timedelta(minutes=5)
    __table_args__ = (Index('ix_statistics_short_term_statistic_id_start', 'metadata_id', 'start', unique=True),)
    __tablename__ = TABLE_STATISTICS_SHORT_TERM

class StatisticsMeta(Base):
    """Statistics meta data."""
    __table_args__ = ({'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},)
    __tablename__ = TABLE_STATISTICS_META
    id = Column(Integer, Identity(), primary_key=True)
    statistic_id = Column(String(255), index=True, unique=True)
    source = Column(String(32))
    unit_of_measurement = Column(String(255))
    has_mean = Column(Boolean)
    has_sum = Column(Boolean)
    name = Column(String(255))

    @staticmethod
    def from_meta(meta: StatisticMetaData) -> StatisticsMeta:
        if False:
            while True:
                i = 10
        'Create object from meta data.'
        return StatisticsMeta(**meta)

class RecorderRuns(Base):
    """Representation of recorder run."""
    __table_args__ = (Index('ix_recorder_runs_start_end', 'start', 'end'),)
    __tablename__ = TABLE_RECORDER_RUNS
    run_id = Column(Integer, Identity(), primary_key=True)
    start = Column(DATETIME_TYPE, default=dt_util.utcnow)
    end = Column(DATETIME_TYPE)
    closed_incorrect = Column(Boolean, default=False)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Return string representation of instance for debugging.'
        end = f"'{self.end.isoformat(sep=' ', timespec='seconds')}'" if self.end else None
        return f"<recorder.RecorderRuns(id={self.run_id}, start='{self.start.isoformat(sep=' ', timespec='seconds')}', end={end}, closed_incorrect={self.closed_incorrect}, created='{self.created.isoformat(sep=' ', timespec='seconds')}')>"

    def entity_ids(self, point_in_time: datetime | None=None) -> list[str]:
        if False:
            while True:
                i = 10
        'Return the entity ids that existed in this run.\n\n        Specify point_in_time if you want to know which existed at that point\n        in time inside the run.\n        '
        session = Session.object_session(self)
        assert session is not None, 'RecorderRuns need to be persisted'
        query = session.query(distinct(States.entity_id)).filter(States.last_updated >= self.start)
        if point_in_time is not None:
            query = query.filter(States.last_updated < point_in_time)
        elif self.end is not None:
            query = query.filter(States.last_updated < self.end)
        return [row[0] for row in query]

    def to_native(self, validate_entity_id: bool=True) -> RecorderRuns:
        if False:
            for i in range(10):
                print('nop')
        'Return self, native format is this model.'
        return self

class SchemaChanges(Base):
    """Representation of schema version changes."""
    __tablename__ = TABLE_SCHEMA_CHANGES
    change_id = Column(Integer, Identity(), primary_key=True)
    schema_version = Column(Integer)
    changed = Column(DATETIME_TYPE, default=dt_util.utcnow)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Return string representation of instance for debugging.'
        return f"<recorder.SchemaChanges(id={self.change_id}, schema_version={self.schema_version}, changed='{self.changed.isoformat(sep=' ', timespec='seconds')}')>"

class StatisticsRuns(Base):
    """Representation of statistics run."""
    __tablename__ = TABLE_STATISTICS_RUNS
    run_id = Column(Integer, Identity(), primary_key=True)
    start = Column(DATETIME_TYPE, index=True)

    def __repr__(self) -> str:
        if False:
            return 10
        'Return string representation of instance for debugging.'
        return f"<recorder.StatisticsRuns(id={self.run_id}, start='{self.start.isoformat(sep=' ', timespec='seconds')}', )>"
EVENT_DATA_JSON = type_coerce(EventData.shared_data.cast(JSONB_VARIANT_CAST), JSONLiteral(none_as_null=True))
OLD_FORMAT_EVENT_DATA_JSON = type_coerce(Events.event_data.cast(JSONB_VARIANT_CAST), JSONLiteral(none_as_null=True))
SHARED_ATTRS_JSON = type_coerce(StateAttributes.shared_attrs.cast(JSON_VARIANT_CAST), JSON(none_as_null=True))
OLD_FORMAT_ATTRS_JSON = type_coerce(States.attributes.cast(JSON_VARIANT_CAST), JSON(none_as_null=True))
ENTITY_ID_IN_EVENT: Column = EVENT_DATA_JSON['entity_id']
OLD_ENTITY_ID_IN_EVENT: Column = OLD_FORMAT_EVENT_DATA_JSON['entity_id']
DEVICE_ID_IN_EVENT: Column = EVENT_DATA_JSON['device_id']
OLD_STATE = aliased(States, name='old_state')

@overload
def process_timestamp(ts: None) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def process_timestamp(ts: datetime) -> datetime:
    if False:
        return 10
    ...

def process_timestamp(ts: datetime | None) -> datetime | None:
    if False:
        while True:
            i = 10
    'Process a timestamp into datetime object.'
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt_util.UTC)
    return dt_util.as_utc(ts)