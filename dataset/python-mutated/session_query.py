from typing import Dict, Optional, Tuple, Union
from posthog.models import Filter
from posthog.models.filters.path_filter import PathFilter
from posthog.models.filters.retention_filter import RetentionFilter
from posthog.models.filters.stickiness_filter import StickinessFilter
from posthog.models.team import Team
from posthog.queries.query_date_range import QueryDateRange

class SessionQuery:
    """
    Query class responsible for creating and joining sessions
    """
    SESSION_TABLE_ALIAS = 'sessions'
    _filter: Union[Filter, PathFilter, RetentionFilter, StickinessFilter]
    _team_id: int
    _session_id_alias: Optional[str]

    def __init__(self, filter: Union[Filter, PathFilter, RetentionFilter, StickinessFilter], team: Team, session_id_alias=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._filter = filter
        self._team = team
        self._session_id_alias = session_id_alias

    def get_query(self) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        params = {'team_id': self._team.pk}
        query_date_range = QueryDateRange(filter=self._filter, team=self._team, should_round=False)
        (parsed_date_from, date_from_params) = query_date_range.date_from
        (parsed_date_to, date_to_params) = query_date_range.date_to
        params.update(date_from_params)
        params.update(date_to_params)
        return (f'''\n                SELECT\n                    "$session_id"{(f' AS {self._session_id_alias}' if self._session_id_alias else '')},\n                    dateDiff('second',min(timestamp), max(timestamp)) as session_duration\n                FROM\n                    events\n                WHERE\n                    {self._session_id_alias or '"$session_id"'} != ''\n                    AND team_id = %(team_id)s\n                    {parsed_date_from} - INTERVAL 24 HOUR\n                    {parsed_date_to} + INTERVAL 24 HOUR\n                GROUP BY {self._session_id_alias or '"$session_id"'}\n            ''', params)

    @property
    def is_used(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether any columns from session are actually being queried'
        if not isinstance(self._filter, StickinessFilter) and self._filter.breakdown_type == 'session':
            return True
        if any((prop.type == 'session' for prop in self._filter.property_groups.flat)):
            return True
        if any((prop.type == 'session' for entity in self._filter.entities for prop in entity.property_groups.flat)):
            return True
        if any((entity.math_property == '$session_duration' for entity in self._filter.entities)):
            return True
        return False