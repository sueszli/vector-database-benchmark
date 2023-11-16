from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Tuple
from posthog.client import sync_execute
from posthog.models.event.util import parse_properties
from posthog.models.filters.session_recordings_filter import SessionRecordingsFilter
from posthog.queries.event_query import EventQuery
if TYPE_CHECKING:
    from posthog.models import Team

class EventFiltersSQL(NamedTuple):
    aggregate_select_clause: str
    aggregate_having_clause: str
    where_conditions: str
    params: Dict[str, Any]

class SessionRecordingProperties(EventQuery):
    _filter: SessionRecordingsFilter
    _session_ids: List[str]
    SESSION_RECORDING_PROPERTIES_ALLOWLIST = {'$os', '$browser', '$device_type', '$current_url', '$host', '$pathname', '$geoip_country_code', '$geoip_country_name'}
    _core_single_pageview_event_query = '\n         SELECT\n            "$session_id" AS session_id,\n            any(properties) AS properties\n         FROM events\n         PREWHERE\n             team_id = %(team_id)s\n             AND event IN [\'$pageview\', \'$autocapture\']\n             {session_ids_clause}\n             {events_timestamp_clause}\n             GROUP BY session_id\n    '

    def __init__(self, team: 'Team', session_ids: List[str], filter: SessionRecordingsFilter):
        if False:
            i = 10
            return i + 15
        super().__init__(team=team, filter=filter)
        self._session_ids = sorted(session_ids)

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        self._should_join_distinct_ids = False

    def _get_events_timestamp_clause(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        timestamp_clause = ''
        timestamp_params = {}
        if self._filter.date_from:
            timestamp_clause += '\nAND timestamp >= %(event_start_time)s'
            timestamp_params['event_start_time'] = self._filter.date_from - timedelta(hours=12)
        if self._filter.date_to:
            timestamp_clause += '\nAND timestamp <= %(event_end_time)s'
            timestamp_params['event_end_time'] = self._filter.date_to + timedelta(hours=12)
        return (timestamp_clause, timestamp_params)

    def format_session_recording_id_filters(self) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        where_conditions = 'AND session_id IN %(session_ids)s'
        return (where_conditions, {'session_ids': self._session_ids})

    def get_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        base_params = {'team_id': self._team_id}
        (events_timestamp_clause, events_timestamp_params) = self._get_events_timestamp_clause()
        (session_ids_clause, session_ids_params) = self.format_session_recording_id_filters()
        return (self._core_single_pageview_event_query.format(events_timestamp_clause=events_timestamp_clause, session_ids_clause=session_ids_clause), {**base_params, **events_timestamp_params, **session_ids_params})

    def _data_to_return(self, results: List[Any]) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        return [{'session_id': row[0], 'properties': parse_properties(row[1], self.SESSION_RECORDING_PROPERTIES_ALLOWLIST)} for row in results]

    def run(self) -> List:
        if False:
            i = 10
            return i + 15
        (query, query_params) = self.get_query()
        query_results = sync_execute(query, query_params)
        session_recording_properties = self._data_to_return(query_results)
        return session_recording_properties