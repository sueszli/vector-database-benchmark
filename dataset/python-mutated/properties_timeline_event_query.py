import datetime as dt
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo
from posthog.models.entity.util import get_entity_filtering_params
from posthog.models.filters.properties_timeline_filter import PropertiesTimelineFilter
from posthog.queries.event_query import EventQuery
from posthog.queries.query_date_range import QueryDateRange
from posthog.queries.util import PersonPropertiesMode

class PropertiesTimelineEventQuery(EventQuery):
    effective_date_from: dt.datetime
    effective_date_to: dt.datetime
    _filter: PropertiesTimelineFilter
    _group_type_index: Optional[int]

    def __init__(self, filter: PropertiesTimelineFilter, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(filter, *args, **kwargs)
        self._group_type_index = filter.aggregation_group_type_index

    def get_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        real_fields = [f'{self.EVENT_TABLE_ALIAS}.timestamp AS timestamp']
        sentinel_fields = ['NULL AS timestamp']
        if self._group_type_index is None:
            columns_to_query = self._column_optimizer.person_on_event_columns_to_query | {'person_properties'}
        else:
            columns_to_query = self._column_optimizer.group_on_event_columns_to_query | {f'group_{self._group_type_index}_properties'}
        for column_name in sorted(columns_to_query):
            real_fields.append(f'{self.EVENT_TABLE_ALIAS}."{column_name}" AS "{column_name}"')
            sentinel_fields.append(f''''' AS "{column_name}"''')
        real_fields_combined = ',\n'.join(real_fields)
        sentinel_fields_combined = ',\n'.join(sentinel_fields)
        (date_query, date_params) = self._get_date_filter()
        self.params.update(date_params)
        (entity_query, entity_params) = self._get_entity_query()
        self.params.update(entity_params)
        actor_id_column = 'person_id' if self._group_type_index is None else f'$group_{self._group_type_index}'
        query = f'\n            (\n                SELECT {real_fields_combined}\n                FROM events {self.EVENT_TABLE_ALIAS}\n                WHERE\n                    team_id = %(team_id)s\n                    AND {actor_id_column} = %(actor_id)s\n                    {entity_query}\n                    {date_query}\n                ORDER BY timestamp ASC\n            ) UNION ALL (\n                SELECT {sentinel_fields_combined} /* We need a final sentinel row for relevant_event_count */\n            )\n        '
        return (query, self.params)

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            while True:
                i = 10
        self._should_join_distinct_ids = False

    def _determine_should_join_persons(self) -> None:
        if False:
            print('Hello World!')
        self._should_join_persons = False

    def _determine_should_join_sessions(self) -> None:
        if False:
            while True:
                i = 10
        self._should_join_sessions = False

    def _get_date_filter(self) -> Tuple[str, Dict]:
        if False:
            i = 10
            return i + 15
        query_params: Dict[str, Any] = {}
        query_date_range = QueryDateRange(self._filter, self._team)
        effective_timezone = ZoneInfo(self._team.timezone)
        self.effective_date_from = query_date_range.date_from_param.replace(tzinfo=effective_timezone)
        self.effective_date_to = query_date_range.date_to_param.replace(tzinfo=effective_timezone)
        (parsed_date_from, date_from_params) = query_date_range.date_from
        (parsed_date_to, date_to_params) = query_date_range.date_to
        query_params.update(date_from_params)
        query_params.update(date_to_params)
        date_filter = f'{parsed_date_from} {parsed_date_to}'
        return (date_filter, query_params)

    def _get_entity_query(self) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        (entity_params, entity_format_params) = get_entity_filtering_params(allowed_entities=self._filter.entities, team_id=self._team_id, table_name=self.EVENT_TABLE_ALIAS, person_properties_mode=PersonPropertiesMode.DIRECT_ON_EVENTS, hogql_context=self._filter.hogql_context)
        return (entity_format_params.get('entity_query', ''), entity_params)