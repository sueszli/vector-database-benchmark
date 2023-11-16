"""Models for SQLAlchemy.

This file contains the model definitions for schema version 16,
used by Home Assistant Core 2021.6.0, which was the initial version
to include long term statistics.
It is used to test the schema migration logic.
"""
import json
import logging
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer, String, Text, distinct
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm.session import Session
from homeassistant.const import MAX_LENGTH_EVENT_CONTEXT_ID, MAX_LENGTH_EVENT_EVENT_TYPE, MAX_LENGTH_EVENT_ORIGIN, MAX_LENGTH_STATE_DOMAIN, MAX_LENGTH_STATE_ENTITY_ID, MAX_LENGTH_STATE_STATE
from homeassistant.core import Context, Event, EventOrigin, State, split_entity_id
from homeassistant.helpers.json import JSONEncoder
import homeassistant.util.dt as dt_util
Base = declarative_base()
SCHEMA_VERSION = 16
_LOGGER = logging.getLogger(__name__)
DB_TIMEZONE = '+00:00'
TABLE_EVENTS = 'events'
TABLE_STATES = 'states'
TABLE_RECORDER_RUNS = 'recorder_runs'
TABLE_SCHEMA_CHANGES = 'schema_changes'
TABLE_STATISTICS = 'statistics'
ALL_TABLES = [TABLE_STATES, TABLE_EVENTS, TABLE_RECORDER_RUNS, TABLE_SCHEMA_CHANGES, TABLE_STATISTICS]
DATETIME_TYPE = DateTime(timezone=True).with_variant(mysql.DATETIME(timezone=True, fsp=6), 'mysql')

class Events(Base):
    """Event history data."""
    __table_args__ = {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}
    __tablename__ = TABLE_EVENTS
    event_id = Column(Integer, Identity(), primary_key=True)
    event_type = Column(String(MAX_LENGTH_EVENT_EVENT_TYPE))
    event_data = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    origin = Column(String(MAX_LENGTH_EVENT_ORIGIN))
    time_fired = Column(DATETIME_TYPE, index=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    context_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_user_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_parent_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    __table_args__ = (Index('ix_events_event_type_time_fired', 'event_type', 'time_fired'),)

    def __repr__(self) -> str:
        if False:
            return 10
        'Return string representation of instance for debugging.'
        return f"<recorder.Events(id={self.event_id}, type='{self.event_type}', data='{self.event_data}', origin='{self.origin}', time_fired='{self.time_fired}')>"

    @staticmethod
    def from_event(event, event_data=None):
        if False:
            while True:
                i = 10
        'Create an event database object from a native event.'
        return Events(event_type=event.event_type, event_data=event_data or json.dumps(event.data, cls=JSONEncoder), origin=str(event.origin.value), time_fired=event.time_fired, context_id=event.context.id, context_user_id=event.context.user_id, context_parent_id=event.context.parent_id)

    def to_native(self, validate_entity_id=True):
        if False:
            print('Hello World!')
        'Convert to a natve HA Event.'
        context = Context(id=self.context_id, user_id=self.context_user_id, parent_id=self.context_parent_id)
        try:
            return Event(self.event_type, json.loads(self.event_data), EventOrigin(self.origin), process_timestamp(self.time_fired), context=context)
        except ValueError:
            _LOGGER.exception('Error converting to event: %s', self)
            return None

class States(Base):
    """State change history."""
    __table_args__ = {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}
    __tablename__ = TABLE_STATES
    state_id = Column(Integer, Identity(), primary_key=True)
    domain = Column(String(MAX_LENGTH_STATE_DOMAIN))
    entity_id = Column(String(MAX_LENGTH_STATE_ENTITY_ID))
    state = Column(String(MAX_LENGTH_STATE_STATE))
    attributes = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    event_id = Column(Integer, ForeignKey('events.event_id', ondelete='CASCADE'), index=True)
    last_changed = Column(DATETIME_TYPE, default=dt_util.utcnow)
    last_updated = Column(DATETIME_TYPE, default=dt_util.utcnow, index=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    old_state_id = Column(Integer, ForeignKey('states.state_id'), index=True)
    event = relationship('Events', uselist=False)
    old_state = relationship('States', remote_side=[state_id])
    __table_args__ = (Index('ix_states_entity_id_last_updated', 'entity_id', 'last_updated'),)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return string representation of instance for debugging.'
        return f"<recorder.States(id={self.state_id}, domain='{self.domain}', entity_id='{self.entity_id}', state='{self.state}', event_id='{self.event_id}', last_updated='{self.last_updated.isoformat(sep=' ', timespec='seconds')}', old_state_id={self.old_state_id})>"

    @staticmethod
    def from_event(event):
        if False:
            i = 10
            return i + 15
        'Create object from a state_changed event.'
        entity_id = event.data['entity_id']
        state = event.data.get('new_state')
        dbstate = States(entity_id=entity_id)
        if state is None:
            dbstate.state = ''
            dbstate.domain = split_entity_id(entity_id)[0]
            dbstate.attributes = '{}'
            dbstate.last_changed = event.time_fired
            dbstate.last_updated = event.time_fired
        else:
            dbstate.domain = state.domain
            dbstate.state = state.state
            dbstate.attributes = json.dumps(dict(state.attributes), cls=JSONEncoder)
            dbstate.last_changed = state.last_changed
            dbstate.last_updated = state.last_updated
        return dbstate

    def to_native(self, validate_entity_id=True):
        if False:
            return 10
        'Convert to an HA state object.'
        try:
            return State(self.entity_id, self.state, json.loads(self.attributes), process_timestamp(self.last_changed), process_timestamp(self.last_updated), context=Context(id=None), validate_entity_id=validate_entity_id)
        except ValueError:
            _LOGGER.exception('Error converting row to state: %s', self)
            return None

class Statistics(Base):
    """Statistics."""
    __table_args__ = {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}
    __tablename__ = TABLE_STATISTICS
    id = Column(Integer, primary_key=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    source = Column(String(32))
    statistic_id = Column(String(255))
    start = Column(DATETIME_TYPE, index=True)
    mean = Column(Float())
    min = Column(Float())
    max = Column(Float())
    last_reset = Column(DATETIME_TYPE)
    state = Column(Float())
    sum = Column(Float())
    __table_args__ = (Index('ix_statistics_statistic_id_start', 'statistic_id', 'start'),)

    @staticmethod
    def from_stats(source, statistic_id, start, stats):
        if False:
            print('Hello World!')
        'Create object from a statistics.'
        return Statistics(source=source, statistic_id=statistic_id, start=start, **stats)

class RecorderRuns(Base):
    """Representation of recorder run."""
    __tablename__ = TABLE_RECORDER_RUNS
    run_id = Column(Integer, Identity(), primary_key=True)
    start = Column(DateTime(timezone=True), default=dt_util.utcnow)
    end = Column(DateTime(timezone=True))
    closed_incorrect = Column(Boolean, default=False)
    created = Column(DateTime(timezone=True), default=dt_util.utcnow)
    __table_args__ = (Index('ix_recorder_runs_start_end', 'start', 'end'),)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return string representation of instance for debugging.'
        end = f"'{self.end.isoformat(sep=' ', timespec='seconds')}'" if self.end else None
        return f"<recorder.RecorderRuns(id={self.run_id}, start='{self.start.isoformat(sep=' ', timespec='seconds')}', end={end}, closed_incorrect={self.closed_incorrect}, created='{self.created.isoformat(sep=' ', timespec='seconds')}')>"

    def entity_ids(self, point_in_time=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the entity ids that existed in this run.\n\n        Specify point_in_time if you want to know which existed at that point\n        in time inside the run.\n        '
        session = Session.object_session(self)
        assert session is not None, 'RecorderRuns need to be persisted'
        query = session.query(distinct(States.entity_id)).filter(States.last_updated >= self.start)
        if point_in_time is not None:
            query = query.filter(States.last_updated < point_in_time)
        elif self.end is not None:
            query = query.filter(States.last_updated < self.end)
        return [row[0] for row in query]

    def to_native(self, validate_entity_id=True):
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
    changed = Column(DateTime(timezone=True), default=dt_util.utcnow)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return string representation of instance for debugging.'
        return f"<recorder.SchemaChanges(id={self.change_id}, schema_version={self.schema_version}, changed='{self.changed.isoformat(sep=' ', timespec='seconds')}')>"

def process_timestamp(ts):
    if False:
        print('Hello World!')
    'Process a timestamp into datetime object.'
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt_util.UTC)
    return dt_util.as_utc(ts)

def process_timestamp_to_utc_isoformat(ts):
    if False:
        return 10
    'Process a timestamp into UTC isotime.'
    if ts is None:
        return None
    if ts.tzinfo == dt_util.UTC:
        return ts.isoformat()
    if ts.tzinfo is None:
        return f'{ts.isoformat()}{DB_TIMEZONE}'
    return ts.astimezone(dt_util.UTC).isoformat()

class LazyState(State):
    """A lazy version of core State."""
    __slots__ = ['_row', 'entity_id', 'state', '_attributes', '_last_changed', '_last_updated', '_context']

    def __init__(self, row):
        if False:
            print('Hello World!')
        'Init the lazy state.'
        self._row = row
        self.entity_id = self._row.entity_id
        self.state = self._row.state or ''
        self._attributes = None
        self._last_changed = None
        self._last_updated = None
        self._context = None

    @property
    def attributes(self):
        if False:
            i = 10
            return i + 15
        'State attributes.'
        if not self._attributes:
            try:
                self._attributes = json.loads(self._row.attributes)
            except ValueError:
                _LOGGER.exception('Error converting row to state: %s', self._row)
                self._attributes = {}
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        if False:
            i = 10
            return i + 15
        'Set attributes.'
        self._attributes = value

    @property
    def context(self):
        if False:
            i = 10
            return i + 15
        'State context.'
        if not self._context:
            self._context = Context(id=None)
        return self._context

    @context.setter
    def context(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Set context.'
        self._context = value

    @property
    def last_changed(self):
        if False:
            i = 10
            return i + 15
        'Last changed datetime.'
        if not self._last_changed:
            self._last_changed = process_timestamp(self._row.last_changed)
        return self._last_changed

    @last_changed.setter
    def last_changed(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Set last changed datetime.'
        self._last_changed = value

    @property
    def last_updated(self):
        if False:
            i = 10
            return i + 15
        'Last updated datetime.'
        if not self._last_updated:
            self._last_updated = process_timestamp(self._row.last_updated)
        return self._last_updated

    @last_updated.setter
    def last_updated(self, value):
        if False:
            while True:
                i = 10
        'Set last updated datetime.'
        self._last_updated = value

    def as_dict(self):
        if False:
            while True:
                i = 10
        'Return a dict representation of the LazyState.\n\n        Async friendly.\n        To be used for JSON serialization.\n        '
        if self._last_changed:
            last_changed_isoformat = self._last_changed.isoformat()
        else:
            last_changed_isoformat = process_timestamp_to_utc_isoformat(self._row.last_changed)
        if self._last_updated:
            last_updated_isoformat = self._last_updated.isoformat()
        else:
            last_updated_isoformat = process_timestamp_to_utc_isoformat(self._row.last_updated)
        return {'entity_id': self.entity_id, 'state': self.state, 'attributes': self._attributes or self.attributes, 'last_changed': last_changed_isoformat, 'last_updated': last_updated_isoformat}

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return the comparison.'
        return other.__class__ in [self.__class__, State] and self.entity_id == other.entity_id and (self.state == other.state) and (self.attributes == other.attributes)