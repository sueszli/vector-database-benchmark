import dataclasses
from typing import Any, Dict, List, Optional, Tuple
from posthog.models.filters.retention_filter import RetentionFilter
from posthog.models.team import Team
from posthog.queries.actor_base_query import ActorBaseQuery
from posthog.queries.insight import insight_sync_execute
from posthog.queries.retention.retention_events_query import RetentionEventsQuery
from posthog.queries.retention.sql import RETENTION_BREAKDOWN_ACTOR_SQL
from posthog.queries.retention.types import BreakdownValues

@dataclasses.dataclass
class AppearanceRow:
    """
    Container for the rows of the "Appearance count" query.
    """
    actor_id: str
    appearance_count: int
    appearances: List[float]

class RetentionActorsByPeriod(ActorBaseQuery):
    _filter: RetentionFilter
    _retention_events_query = RetentionEventsQuery
    QUERY_TYPE = 'retention_actors_by_period'

    def __init__(self, team: Team, filter: RetentionFilter):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(team, filter)

    def actors(self):
        if False:
            while True:
                i = 10
        '\n        Creates a response of the form\n\n        ```\n        [\n            {\n                "person": {"distinct_id": ..., ...},\n                "appearance_count": 3,\n                "appearances": [1, 0, 1, 1, 0, 0]\n            }\n            ...\n        ]\n        ```\n\n        where appearances values represent if the person was active in an\n        interval, where the index of the list is the interval it refers to.\n        '
        (actor_query, actor_query_params) = _build_actor_query(filter=self._filter, team=self._team, filter_by_breakdown=self._filter.breakdown_values or (self._filter.selected_interval,) if self._filter.selected_interval is not None else None, retention_events_query=self._retention_events_query)
        results = insight_sync_execute(actor_query, {**actor_query_params, **self._filter.hogql_context.values}, query_type='retention_actors', filter=self._filter, team_id=self._team.pk)
        actor_appearances = [AppearanceRow(actor_id=str(row[0]), appearance_count=len(row[1]), appearances=row[1]) for row in results]
        (_, serialized_actors) = self.get_actors_from_result([(actor_appearance.actor_id,) for actor_appearance in actor_appearances])
        actors_lookup = {str(actor['id']): actor for actor in serialized_actors}
        return ([{'person': actors_lookup[actor.actor_id], 'appearances': [1 if interval_number in actor.appearances else 0 for interval_number in range(self._filter.total_intervals - (self._filter.selected_interval or 0))]} for actor in actor_appearances if actor.actor_id in actors_lookup], len(actor_appearances))

def build_actor_activity_query(filter: RetentionFilter, team: Team, filter_by_breakdown: Optional[BreakdownValues]=None, selected_interval: Optional[int]=None, aggregate_users_by_distinct_id: Optional[bool]=None, retention_events_query=RetentionEventsQuery) -> Tuple[str, Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    from posthog.queries.retention import build_returning_event_query, build_target_event_query
    '\n    The retention actor query is used to retrieve something of the form:\n\n        breakdown_values, intervals_from_base, actor_id\n\n    We use actor here as an abstraction over the different types we can have aside from\n    person_ids\n    '
    (returning_event_query, returning_event_query_params) = build_returning_event_query(filter=filter, team=team, aggregate_users_by_distinct_id=aggregate_users_by_distinct_id, person_on_events_mode=team.person_on_events_mode, retention_events_query=retention_events_query)
    (target_event_query, target_event_query_params) = build_target_event_query(filter=filter, team=team, aggregate_users_by_distinct_id=aggregate_users_by_distinct_id, person_on_events_mode=team.person_on_events_mode, retention_events_query=retention_events_query)
    all_params = {'period': filter.period.lower(), 'breakdown_values': list(filter_by_breakdown) if filter_by_breakdown else None, 'selected_interval': selected_interval, **returning_event_query_params, **target_event_query_params}
    query = RETENTION_BREAKDOWN_ACTOR_SQL.format(returning_event_query=returning_event_query, target_event_query=target_event_query)
    return (query, all_params)

def _build_actor_query(filter: RetentionFilter, team: Team, filter_by_breakdown: Optional[BreakdownValues]=None, selected_interval: Optional[int]=None, retention_events_query=RetentionEventsQuery) -> Tuple[str, Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    (actor_activity_query, actor_activity_query_params) = build_actor_activity_query(filter=filter, team=team, filter_by_breakdown=filter_by_breakdown, selected_interval=selected_interval, aggregate_users_by_distinct_id=False, retention_events_query=retention_events_query)
    params = {'offset': filter.offset, 'limit': filter.limit or 100, **actor_activity_query_params}
    actor_query_template = '\n        SELECT\n            actor_id,\n            groupArray(actor_activity.intervals_from_base) AS appearances\n\n        FROM ({actor_activity_query}) AS actor_activity\n\n        GROUP BY actor_id\n\n        -- make sure we have stable ordering/pagination\n        -- NOTE: relies on ids being monotonic\n        ORDER BY length(appearances) DESC, actor_id\n\n        LIMIT %(limit)s\n        OFFSET %(offset)s\n    '
    actor_query = actor_query_template.format(actor_activity_query=actor_activity_query)
    return (actor_query, params)