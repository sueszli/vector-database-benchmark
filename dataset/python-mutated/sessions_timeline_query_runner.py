from datetime import timedelta
import json
from typing import Dict, Optional, Any, cast
from posthog.api.element import ElementSerializer
from posthog.clickhouse.client.connection import Workload
from posthog.hogql import ast
from posthog.hogql.parser import parse_select
from posthog.hogql.query import execute_hogql_query
from posthog.hogql.timings import HogQLTimings
from posthog.hogql_queries.query_runner import QueryRunner
from posthog.models import Team
from posthog.models.element.element import chain_to_elements
from posthog.schema import EventType, SessionsTimelineQuery, SessionsTimelineQueryResponse, TimelineEntry
from posthog.utils import relative_date_parse

class SessionsTimelineQueryRunner(QueryRunner):
    """
    ## How does the sessions timeline work?

    A formal session on the timeline is defined as a collection of all events with a given session ID.
    An informal session is defined as a collection of contiguous events that don't have a session ID.
    Additionally, a new informal session is formed when the time between two consecutive events exceeds 30 minutes
    (which does not apply to formal sessions).

    > Note that the logic above is not the same as that of Trends session duration.
    > In Trends, only events with a session ID are considered (i.e. formal sessions).

    Now, the sessions timeline is a sequence of sessions (both formal and informal), starting with ones that started
    most recently. Events within a session are also ordered with latest first.
    """
    EVENT_LIMIT = 1000
    query: SessionsTimelineQuery
    query_type = SessionsTimelineQuery

    def __init__(self, query: SessionsTimelineQuery | Dict[str, Any], team: Team, timings: Optional[HogQLTimings]=None):
        if False:
            while True:
                i = 10
        super().__init__(query, team, timings)
        if isinstance(query, SessionsTimelineQuery):
            self.query = query
        else:
            self.query = SessionsTimelineQuery.model_validate(query)

    def _get_events_subquery(self) -> ast.SelectQuery:
        if False:
            for i in range(10):
                print('nop')
        after = relative_date_parse(self.query.after or '-24h', self.team.timezone_info)
        before = relative_date_parse(self.query.before or '-0h', self.team.timezone_info)
        with self.timings.measure('build_events_subquery'):
            event_conditions: list[ast.Expr] = [ast.CompareOperation(op=ast.CompareOperationOp.Gt, left=ast.Field(chain=['timestamp']), right=ast.Constant(value=after)), ast.CompareOperation(op=ast.CompareOperationOp.Lt, left=ast.Field(chain=['timestamp']), right=ast.Constant(value=before))]
            if self.query.personId:
                event_conditions.append(ast.CompareOperation(left=ast.Field(chain=['person_id']), right=ast.Constant(value=self.query.personId), op=ast.CompareOperationOp.Eq))
            select_query = parse_select('\n                SELECT\n                    uuid,\n                    person_id AS person_id,\n                    timestamp AS timestamp,\n                    event,\n                    properties,\n                    distinct_id,\n                    elements_chain,\n                    $session_id AS session_id,\n                    lagInFrame($session_id, 1) OVER (\n                        PARTITION BY person_id ORDER BY timestamp\n                    ) AS prev_session_id\n                FROM events\n                WHERE {event_conditions}\n                ORDER BY timestamp DESC\n                LIMIT {event_limit_with_more}', placeholders={'event_limit_with_more': ast.Constant(value=self.EVENT_LIMIT + 1), 'event_conditions': ast.And(exprs=event_conditions)})
        return cast(ast.SelectQuery, select_query)

    def to_query(self) -> ast.SelectQuery:
        if False:
            for i in range(10):
                print('nop')
        if self.timings is None:
            self.timings = HogQLTimings()
        with self.timings.measure('build_sessions_timeline_query'):
            select_query = parse_select("\n                SELECT\n                    e.uuid,\n                    e.timestamp,\n                    e.event,\n                    e.properties,\n                    e.distinct_id,\n                    e.elements_chain,\n                    e.session_id AS formal_session_id,\n                    first_value(e.uuid) OVER (\n                        PARTITION BY (e.person_id, session_id_flip_index) ORDER BY _toInt64(timestamp)\n                        RANGE BETWEEN 1800 PRECEDING AND CURRENT ROW /* split informal session after 30+ min */\n                    ) AS informal_session_uuid,\n                    dateDiff('s', sre.start_time, sre.end_time) AS recording_duration_s\n                FROM (\n                    SELECT\n                        *,\n                        sum(session_id = prev_session_id ? 0 : 1) OVER (\n                            PARTITION BY person_id ORDER BY timestamp ROWS UNBOUNDED PRECEDING\n                        ) AS session_id_flip_index\n                    FROM ({events_subquery})\n                ) e\n                LEFT JOIN (\n                    SELECT start_time AS start_time, end_time AS end_time, session_id FROM session_replay_events\n                ) AS sre\n                ON e.session_id = sre.session_id\n                ORDER BY timestamp DESC", placeholders={'events_subquery': self._get_events_subquery()})
        return cast(ast.SelectQuery, select_query)

    def to_persons_query(self):
        if False:
            print('Hello World!')
        return parse_select('SELECT DISTINCT person_id FROM {events_subquery}', {'events_subquery': self._get_events_subquery()})

    def calculate(self) -> SessionsTimelineQueryResponse:
        if False:
            while True:
                i = 10
        query_result = execute_hogql_query(query=self.to_query(), team=self.team, workload=Workload.ONLINE, query_type='SessionsTimelineQuery', timings=self.timings)
        assert query_result.results is not None
        timeline_entries_map: Dict[str, TimelineEntry] = {}
        for (uuid, timestamp_parsed, event, properties_raw, distinct_id, elements_chain, formal_session_id, informal_session_id, recording_duration_s) in reversed(query_result.results[:self.EVENT_LIMIT]):
            entry_id = str(formal_session_id or informal_session_id)
            if entry_id not in timeline_entries_map:
                timeline_entries_map[entry_id] = TimelineEntry(sessionId=formal_session_id or None, events=[], recording_duration_s=recording_duration_s or None)
            timeline_entries_map[entry_id].events.append(EventType(id=str(uuid), distinct_id=distinct_id, event=event, timestamp=timestamp_parsed.isoformat(), properties=json.loads(properties_raw), elements_chain=elements_chain or None, elements=ElementSerializer(chain_to_elements(elements_chain), many=True).data))
        timeline_entries = list(reversed(timeline_entries_map.values()))
        for entry in timeline_entries:
            entry.events.reverse()
        return SessionsTimelineQueryResponse(results=timeline_entries, hasMore=len(query_result.results) > self.EVENT_LIMIT, timings=self.timings.to_list(), hogql=query_result.hogql)

    def _is_stale(self, cached_result_package):
        if False:
            while True:
                i = 10
        return True

    def _refresh_frequency(self):
        if False:
            i = 10
            return i + 15
        return timedelta(minutes=1)