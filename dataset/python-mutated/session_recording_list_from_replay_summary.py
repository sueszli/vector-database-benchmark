import dataclasses
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, NamedTuple, Tuple, Union
from sentry_sdk import capture_exception
from posthog.client import sync_execute
from posthog.constants import TREND_FILTER_TYPE_ACTIONS, PropertyOperatorType
from posthog.models import Entity, Team
from posthog.models.action.util import format_entity_filter
from posthog.models.filters.mixins.utils import cached_property
from posthog.models.filters.session_recordings_filter import SessionRecordingsFilter
from posthog.models.property import PropertyGroup
from posthog.models.property.util import parse_prop_grouped_clauses
from posthog.models.team import PersonOnEventsMode
from posthog.queries.event_query import EventQuery
from posthog.queries.util import PersonPropertiesMode
from posthog.session_recordings.queries.session_replay_events import ttl_days

@dataclasses.dataclass(frozen=True)
class SummaryEventFiltersSQL:
    having_conditions: str
    having_select: str
    where_conditions: str
    params: Dict[str, Any]

class SessionRecordingQueryResult(NamedTuple):
    results: List
    has_more_recording: bool

def _get_recording_start_time_clause(recording_filters: SessionRecordingsFilter) -> Tuple[str, Dict[str, Any]]:
    if False:
        print('Hello World!')
    start_time_clause = ''
    start_time_params = {}
    if recording_filters.date_from:
        start_time_clause += '\nAND start_time >= %(start_time)s'
        start_time_params['start_time'] = recording_filters.date_from
    if recording_filters.date_to:
        start_time_clause += '\nAND start_time <= %(end_time)s'
        start_time_params['end_time'] = recording_filters.date_to
    return (start_time_clause, start_time_params)

def _get_filter_by_log_text_session_ids_clause(team: Team, recording_filters: SessionRecordingsFilter, column_name='session_id') -> Tuple[str, Dict[str, Any]]:
    if False:
        while True:
            i = 10
    if not recording_filters.console_search_query:
        return ('', {})
    (log_query, log_params) = LogQuery(team=team, filter=recording_filters).get_query()
    return (f'AND "{column_name}" in ({log_query}) as log_text_matching', log_params)

def _get_filter_by_provided_session_ids_clause(recording_filters: SessionRecordingsFilter, column_name='session_id') -> Tuple[str, Dict[str, Any]]:
    if False:
        while True:
            i = 10
    if recording_filters.session_ids is None:
        return ('', {})
    return (f'AND "{column_name}" in %(session_ids)s', {'session_ids': recording_filters.session_ids})

class LogQuery:
    _filter: SessionRecordingsFilter
    _team_id: int
    _team: Team

    def __init__(self, team: Team, filter: SessionRecordingsFilter):
        if False:
            while True:
                i = 10
        self._filter = filter
        self._team = team
        self._team_id = team.pk
    _rawQuery = '\n    SELECT distinct log_source_id as session_id\n    FROM log_entries\n    PREWHERE team_id = %(team_id)s\n            AND timestamp >= %(clamped_to_storage_ttl)s\n            AND timestamp <= now()\n            {events_timestamp_clause}\n    WHERE 1=1\n    {console_log_clause}\n    AND positionCaseInsensitive(message, %(console_search_query)s) > 0\n    '

    @property
    def ttl_days(self):
        if False:
            while True:
                i = 10
        return ttl_days(self._team)

    @cached_property
    def _get_events_timestamp_clause(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        timestamp_clause = ''
        timestamp_params = {}
        if self._filter.date_from:
            timestamp_clause += '\nAND timestamp >= %(event_start_time)s'
            timestamp_params['event_start_time'] = self._filter.date_from - timedelta(hours=12)
        if self._filter.date_to:
            timestamp_clause += '\nAND timestamp <= %(event_end_time)s'
            timestamp_params['event_end_time'] = self._filter.date_to + timedelta(hours=12)
        return (timestamp_clause, timestamp_params)

    @staticmethod
    def _get_console_log_clause(console_logs_filter: List[Literal['error', 'warn', 'log']]) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        return (f'AND level in %(console_logs_levels)s', {'console_logs_levels': console_logs_filter}) if console_logs_filter else ('', {})

    def get_query(self) -> Tuple[str, Dict]:
        if False:
            while True:
                i = 10
        if not self._filter.console_search_query:
            return ('', {})
        (events_timestamp_clause, events_timestamp_params) = self._get_events_timestamp_clause
        (console_log_clause, console_log_params) = self._get_console_log_clause(self._filter.console_logs_filter)
        return (self._rawQuery.format(events_timestamp_clause=events_timestamp_clause, console_log_clause=console_log_clause), {'team_id': self._team_id, 'clamped_to_storage_ttl': datetime.now() - timedelta(days=self.ttl_days), 'console_search_query': self._filter.console_search_query, **events_timestamp_params, **console_log_params})

class PersonsQuery(EventQuery):
    _filter: SessionRecordingsFilter

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def _data_to_return(self, results: List[Any]) -> List[Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        pass
    _raw_persons_query = '\n        SELECT distinct_id, argMax(person_id, version) as person_id\n        {select_person_props}\n        FROM person_distinct_id2 as pdi\n            {filter_persons_clause}\n        WHERE team_id = %(team_id)s\n        {prop_filter_clause}\n        GROUP BY distinct_id\n        HAVING\n            argMax(is_deleted, version) = 0\n            {prop_having_clause}\n            {filter_by_person_uuid_condition}\n    '

    def get_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        (prop_query, prop_params) = self._get_prop_groups(PropertyGroup(type=PropertyOperatorType.AND, values=[g for g in self._filter.property_groups.flat if g.type == 'person' or 'cohort' in g.type]), person_id_joined_alias=f'{self.DISTINCT_ID_TABLE_ALIAS}.person_id')
        (having_prop_query, having_prop_params) = self._get_prop_groups(PropertyGroup(type=PropertyOperatorType.AND, values=[g for g in self._filter.property_groups.flat if g.type == 'hogql' and 'person.properties' in g.key]), person_id_joined_alias=f'{self.DISTINCT_ID_TABLE_ALIAS}.person_id')
        (person_query, person_query_params) = self._get_person_query()
        should_join_persons = self._filter.person_uuid or person_query
        if not should_join_persons:
            return ('', {})
        else:
            filter_persons_clause = person_query or ''
            filter_by_person_uuid_condition = 'and person_id = %(person_uuid)s' if self._filter.person_uuid else ''
            return (self._raw_persons_query.format(filter_persons_clause=filter_persons_clause, select_person_props=', argMax(person_props, version) as person_props' if 'person_props' in filter_persons_clause else '', prop_filter_clause=prop_query, prop_having_clause=having_prop_query, filter_by_person_uuid_condition=filter_by_person_uuid_condition), {'team_id': self._team_id, **person_query_params, 'person_uuid': self._filter.person_uuid, **prop_params, **having_prop_params})

class SessionIdEventsQuery(EventQuery):
    _filter: SessionRecordingsFilter

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def _data_to_return(self, results: List[Any]) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        pass

    def _determine_should_join_events(self):
        if False:
            return 10
        filters_by_event_or_action = self._filter.entities and len(self._filter.entities) > 0
        has_event_property_filters = len([pg for pg in self._filter.property_groups.flat if pg.type == 'event' or (pg.type == 'hogql' and re.search('(?<!person\\.)properties\\.', pg.key))]) > 0
        return filters_by_event_or_action or has_event_property_filters

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)

    @property
    def ttl_days(self):
        if False:
            print('Hello World!')
        return ttl_days(self._team)
    _raw_events_query = '\n        SELECT\n            {select_event_ids}\n            {event_filter_having_events_select}\n            `$session_id`\n        FROM events e\n        {groups_query}\n        -- sometimes we have to join on persons so we can access e.g. person_props in filters\n        {persons_join}\n        PREWHERE\n            team_id = %(team_id)s\n            -- regardless of what other filters are applied\n            -- limit by storage TTL\n            AND e.timestamp >= %(clamped_to_storage_ttl)s\n            AND e.timestamp <= now()\n        WHERE\n            notEmpty(`$session_id`)\n            {events_timestamp_clause}\n            {event_filter_where_conditions}\n            {prop_filter_clause}\n            {provided_session_ids_clause}\n            -- other times we can check distinct id against a sub query which should be faster than joining\n            {persons_sub_query}\n        GROUP BY `$session_id`\n        HAVING 1=1 {event_filter_having_events_condition}\n    '

    def format_event_filter(self, entity: Entity, prepend: str, team_id: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        (filter_sql, params) = format_entity_filter(team_id=team_id, entity=entity, prepend=prepend, filter_by_team=False, person_id_joined_alias=f'{self.DISTINCT_ID_TABLE_ALIAS}.person_id', hogql_context=self._filter.hogql_context)
        (filters, filter_params) = parse_prop_grouped_clauses(team_id=team_id, property_group=entity.property_groups, prepend=prepend, allow_denormalized_props=True, has_person_id_joined=True, person_properties_mode=PersonPropertiesMode.USING_PERSON_PROPERTIES_COLUMN, hogql_context=self._filter.hogql_context)
        filter_sql += f' {filters}'
        params = {**params, **filter_params}
        return (filter_sql, params)

    @cached_property
    def build_event_filters(self) -> SummaryEventFiltersSQL:
        if False:
            print('Hello World!')
        event_names_to_filter: List[Union[int, str]] = []
        params: Dict = {}
        condition_sql = ''
        for (index, entity) in enumerate(self._filter.entities):
            if entity.type == TREND_FILTER_TYPE_ACTIONS:
                action = entity.get_action()
                event_names_to_filter.extend([ae for ae in action.get_step_events() if ae not in event_names_to_filter])
            elif entity.id and entity.id not in event_names_to_filter:
                event_names_to_filter.append(entity.id)
            (this_entity_condition_sql, this_entity_filter_params) = self.format_event_filter(entity, prepend=f'event_matcher_{index}', team_id=self._team_id)
            joining = 'OR' if index > 0 else ''
            condition_sql += f'{joining} {this_entity_condition_sql}'
            condition_sql = f'( {condition_sql} )'
            params = {**params, **this_entity_filter_params}
        params = {**params, 'event_names': list(event_names_to_filter)}
        if len(event_names_to_filter) == 0:
            having_conditions = ''
            having_select = ''
        else:
            having_conditions = 'AND hasAll(event_names, %(event_names)s)'
            having_select = '\n                -- select the unique events in this session to support filtering sessions by presence of an event\n                    groupUniqArray(event) as event_names,'
        return SummaryEventFiltersSQL(having_conditions=having_conditions, having_select=having_select, where_conditions=f'AND {condition_sql}' if condition_sql else '', params=params)

    def _get_groups_query(self) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        try:
            from ee.clickhouse.queries.groups_join_query import GroupsJoinQuery
        except ImportError:
            from posthog.queries.groups_join_query import GroupsJoinQuery
        return GroupsJoinQuery(self._filter, self._team_id, self._column_optimizer, person_on_events_mode=self._person_on_events_mode).get_join_query()

    @cached_property
    def _get_events_timestamp_clause(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        timestamp_clause = ''
        timestamp_params = {}
        if self._filter.date_from:
            timestamp_clause += '\nAND timestamp >= %(event_start_time)s'
            timestamp_params['event_start_time'] = self._filter.date_from - timedelta(hours=12)
        if self._filter.date_to:
            timestamp_clause += '\nAND timestamp <= %(event_end_time)s'
            timestamp_params['event_end_time'] = self._filter.date_to + timedelta(hours=12)
        return (timestamp_clause, timestamp_params)

    def get_query(self, select_event_ids: bool=False) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        if not self._determine_should_join_events():
            return ('', {})
        base_params = {'team_id': self._team_id, 'clamped_to_storage_ttl': datetime.now() - timedelta(days=self.ttl_days)}
        (_, recording_start_time_params) = _get_recording_start_time_clause(self._filter)
        (provided_session_ids_clause, provided_session_ids_params) = _get_filter_by_provided_session_ids_clause(recording_filters=self._filter, column_name='$session_id')
        event_filters = self.build_event_filters
        event_filters_params = event_filters.params
        (events_timestamp_clause, events_timestamp_params) = self._get_events_timestamp_clause
        (groups_query, groups_params) = self._get_groups_query()
        (prop_query, prop_params) = self._get_prop_groups(PropertyGroup(type=PropertyOperatorType.AND, values=[g for g in self._filter.property_groups.flat if g.type == 'hogql' and 'person.properties' not in g.key or (g.type != 'hogql' and 'cohort' not in g.type and (g.type != 'person'))]), person_id_joined_alias=f'{self.DISTINCT_ID_TABLE_ALIAS}.person_id')
        (persons_join, persons_select_params, persons_sub_query) = self._persons_join_or_subquery(event_filters, prop_query)
        return (self._raw_events_query.format(select_event_ids='groupArray(uuid) as event_ids,' if select_event_ids else '', event_filter_where_conditions=event_filters.where_conditions, event_filter_having_events_condition=event_filters.having_conditions, event_filter_having_events_select=event_filters.having_select, events_timestamp_clause=events_timestamp_clause, prop_filter_clause=prop_query, provided_session_ids_clause=provided_session_ids_clause, persons_join=persons_join, persons_sub_query=persons_sub_query, groups_query=groups_query), {**base_params, **recording_start_time_params, **provided_session_ids_params, **events_timestamp_params, **event_filters_params, **prop_params, **persons_select_params, **groups_params})

    def _persons_join_or_subquery(self, event_filters, prop_query):
        if False:
            while True:
                i = 10
        (persons_select, persons_select_params) = PersonsQuery(filter=self._filter, team=self._team).get_query()
        persons_join = ''
        persons_sub_query = ''
        if persons_select:
            if 'person_props' in prop_query or 'pdi.person_id' in prop_query or 'person_props' in event_filters.where_conditions:
                persons_join = f'JOIN ({persons_select}) as pdi on pdi.distinct_id = e.distinct_id'
            else:
                persons_sub_query = f'AND e.distinct_id in (select distinct_id from ({persons_select}) as events_persons_sub_query)'
        return (persons_join, persons_select_params, persons_sub_query)

    @cached_property
    def _get_person_id_clause(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        person_id_clause = ''
        person_id_params = {}
        if self._filter.person_uuid:
            person_id_clause = 'AND person_id = %(person_uuid)s'
            person_id_params = {'person_uuid': self._filter.person_uuid}
        return (person_id_clause, person_id_params)

    def matching_events(self) -> List[str]:
        if False:
            return 10
        self._filter.hogql_context.modifiers.personsOnEventsMode = PersonOnEventsMode.DISABLED
        (query, query_params) = self.get_query(select_event_ids=True)
        query_results = sync_execute(query, {**query_params, **self._filter.hogql_context.values})
        results = [row[0] for row in query_results]
        return [item for sublist in results for item in sublist]

class SessionRecordingListFromReplaySummary(EventQuery):

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass
    _filter: SessionRecordingsFilter
    SESSION_RECORDINGS_DEFAULT_LIMIT = 50

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)

    @property
    def ttl_days(self):
        if False:
            print('Hello World!')
        return ttl_days(self._team)
    _session_recordings_query: str = "\n    SELECT\n       s.session_id,\n       any(s.team_id),\n       any(s.distinct_id),\n       min(s.min_first_timestamp) as start_time,\n       max(s.max_last_timestamp) as end_time,\n       dateDiff('SECOND', start_time, end_time) as duration,\n       argMinMerge(s.first_url) as first_url,\n       sum(s.click_count),\n       sum(s.keypress_count),\n       sum(s.mouse_activity_count),\n       sum(s.active_milliseconds)/1000 as active_seconds,\n       duration-active_seconds as inactive_seconds,\n       sum(s.console_log_count) as console_log_count,\n       sum(s.console_warn_count) as console_warn_count,\n       sum(s.console_error_count) as console_error_count\n    FROM session_replay_events s\n    WHERE s.team_id = %(team_id)s\n        -- regardless of what other filters are applied\n        -- limit by storage TTL\n        AND s.min_first_timestamp >= %(clamped_to_storage_ttl)s\n         -- we can filter on the pre-aggregated timestamp columns\n        -- because any not-the-lowest min value is _more_ greater than the min value\n        -- and any not-the-highest max value is _less_ lower than the max value\n        AND s.min_first_timestamp >= %(start_time)s\n        AND s.max_last_timestamp <= %(end_time)s\n        {persons_sub_query}\n        {events_sub_query}\n    {provided_session_ids_clause}\n    {log_matching_session_ids_clause}\n    GROUP BY session_id\n        HAVING 1=1 {duration_clause} {console_log_clause}\n    ORDER BY start_time DESC\n    LIMIT %(limit)s OFFSET %(offset)s\n    "

    @staticmethod
    def _data_to_return(results: List[Any]) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        default_columns = ['session_id', 'team_id', 'distinct_id', 'start_time', 'end_time', 'duration', 'first_url', 'click_count', 'keypress_count', 'mouse_activity_count', 'active_seconds', 'inactive_seconds', 'console_log_count', 'console_warn_count', 'console_error_count']
        return [{**dict(zip(default_columns, row[:len(default_columns)]))} for row in results]

    def _paginate_results(self, session_recordings) -> SessionRecordingQueryResult:
        if False:
            return 10
        more_recordings_available = False
        if len(session_recordings) > self.limit:
            more_recordings_available = True
            session_recordings = session_recordings[0:self.limit]
        return SessionRecordingQueryResult(session_recordings, more_recordings_available)

    def run(self) -> SessionRecordingQueryResult:
        if False:
            return 10
        try:
            self._filter.hogql_context.modifiers.personsOnEventsMode = PersonOnEventsMode.DISABLED
            (query, query_params) = self.get_query()
            query_results = sync_execute(query, {**query_params, **self._filter.hogql_context.values})
            session_recordings = self._data_to_return(query_results)
            return self._paginate_results(session_recordings)
        except Exception as ex:
            capture_exception(ex, tags={'team_id': self._team.pk})
            raise ex

    @property
    def limit(self):
        if False:
            while True:
                i = 10
        return self._filter.limit or self.SESSION_RECORDINGS_DEFAULT_LIMIT

    def get_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        offset = self._filter.offset or 0
        base_params = {'team_id': self._team_id, 'limit': self.limit + 1, 'offset': offset, 'clamped_to_storage_ttl': datetime.now() - timedelta(days=self.ttl_days)}
        (_, recording_start_time_params) = _get_recording_start_time_clause(self._filter)
        (provided_session_ids_clause, provided_session_ids_params) = _get_filter_by_provided_session_ids_clause(recording_filters=self._filter)
        (log_matching_session_ids_clause, log_matching_session_ids_params) = _get_filter_by_log_text_session_ids_clause(team=self._team, recording_filters=self._filter)
        (duration_clause, duration_params) = self.duration_clause(self._filter.duration_type_filter)
        console_log_clause = self._get_console_log_clause(self._filter.console_logs_filter)
        (events_select, events_join_params) = SessionIdEventsQuery(team=self._team, filter=self._filter).get_query()
        if events_select:
            events_select = f'AND s.session_id in (select `$session_id` as session_id from ({events_select}) as session_events_sub_query)'
        (persons_select, persons_select_params) = PersonsQuery(filter=self._filter, team=self._team).get_query()
        if persons_select:
            persons_select = f'AND s.distinct_id in (select distinct_id from ({persons_select}) as session_persons_sub_query)'
        return (self._session_recordings_query.format(duration_clause=duration_clause, provided_session_ids_clause=provided_session_ids_clause, console_log_clause=console_log_clause, persons_sub_query=persons_select, events_sub_query=events_select, log_matching_session_ids_clause=log_matching_session_ids_clause), {**base_params, **events_join_params, **recording_start_time_params, **duration_params, **provided_session_ids_params, **persons_select_params, **log_matching_session_ids_params})

    def duration_clause(self, duration_filter_type: Literal['duration', 'active_seconds', 'inactive_seconds']) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        duration_clause = ''
        duration_params = {}
        if self._filter.recording_duration_filter:
            if self._filter.recording_duration_filter.operator == 'gt':
                operator = '>'
            else:
                operator = '<'
            duration_clause = '\nAND {duration_type} {operator} %(recording_duration)s'.format(duration_type=duration_filter_type, operator=operator)
            duration_params = {'recording_duration': self._filter.recording_duration_filter.value}
        return (duration_clause, duration_params)

    @staticmethod
    def _get_console_log_clause(console_logs_filter: List[Literal['error', 'warn', 'log']]) -> str:
        if False:
            return 10
        filters = [f'console_{log}_count > 0' for log in console_logs_filter]
        return f"AND ({' OR '.join(filters)})" if filters else ''