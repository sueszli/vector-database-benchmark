from datetime import timedelta
from functools import cached_property
from typing import List, Optional, Union
from django.utils.timezone import now
from posthog.client import sync_execute
from posthog.models import Team
from posthog.models.filters.utils import validate_group_type_index
from posthog.models.group_type_mapping import GroupTypeMapping
from posthog.models.property import GroupTypeIndex
from posthog.queries.actor_base_query import SerializedActor, SerializedGroup, SerializedPerson, get_groups, get_people
from posthog.queries.person_distinct_id_query import get_team_distinct_ids_query

class RelatedActorsQuery:
    DISTINCT_ID_TABLE_ALIAS = 'pdi'
    '\n    This query calculates other groups and persons that are related to a person or a group.\n\n    Two actors are considered related if they have had shared events in the past 90 days.\n    '

    def __init__(self, team: Team, group_type_index: Optional[Union[GroupTypeIndex, str]], id: str):
        if False:
            i = 10
            return i + 15
        self.team = team
        self.group_type_index = validate_group_type_index('group_type_index', group_type_index)
        self.id = id

    def run(self) -> List[SerializedActor]:
        if False:
            while True:
                i = 10
        results: List[SerializedActor] = []
        results.extend(self._query_related_people())
        for group_type_mapping in GroupTypeMapping.objects.filter(team_id=self.team.pk):
            results.extend(self._query_related_groups(group_type_mapping.group_type_index))
        return results

    @property
    def is_aggregating_by_groups(self) -> bool:
        if False:
            print('Hello World!')
        return self.group_type_index is not None

    def _query_related_people(self) -> List[SerializedPerson]:
        if False:
            print('Hello World!')
        if not self.is_aggregating_by_groups:
            return []
        person_ids = self._take_first(sync_execute(f'\n            SELECT DISTINCT {self.DISTINCT_ID_TABLE_ALIAS}.person_id\n            FROM events e\n            {self._distinct_ids_join}\n            WHERE team_id = %(team_id)s\n              AND timestamp > %(after)s\n              AND timestamp < %(before)s\n              AND {self._filter_clause}\n            ', self._params))
        (_, serialized_people) = get_people(self.team, person_ids)
        return serialized_people

    def _query_related_groups(self, group_type_index: GroupTypeIndex) -> List[SerializedGroup]:
        if False:
            while True:
                i = 10
        if group_type_index == self.group_type_index:
            return []
        group_ids = self._take_first(sync_execute(f"\n            SELECT DISTINCT $group_{group_type_index} AS group_key\n            FROM events e\n            {('' if self.is_aggregating_by_groups else self._distinct_ids_join)}\n            JOIN (\n                SELECT group_key\n                FROM groups\n                WHERE team_id = %(team_id)s AND group_type_index = %(group_type_index)s\n                GROUP BY group_key\n            ) groups ON $group_{group_type_index} = groups.group_key\n            WHERE team_id = %(team_id)s\n              AND timestamp > %(after)s\n              AND timestamp < %(before)s\n              AND group_key != ''\n              AND {self._filter_clause}\n            ORDER BY group_key\n            ", {**self._params, 'group_type_index': group_type_index}))
        (_, serialize_groups) = get_groups(self.team.pk, group_type_index, group_ids)
        return serialize_groups

    def _take_first(self, rows: List) -> List:
        if False:
            return 10
        return [row[0] for row in rows]

    @property
    def _filter_clause(self):
        if False:
            return 10
        if self.is_aggregating_by_groups:
            return f'$group_{self.group_type_index} = %(id)s'
        else:
            return f'{self.DISTINCT_ID_TABLE_ALIAS}.person_id = %(id)s'

    @property
    def _distinct_ids_join(self):
        if False:
            i = 10
            return i + 15
        return f'JOIN ({get_team_distinct_ids_query(self.team.pk)}) {self.DISTINCT_ID_TABLE_ALIAS} on e.distinct_id = {self.DISTINCT_ID_TABLE_ALIAS}.distinct_id'

    @cached_property
    def _params(self):
        if False:
            for i in range(10):
                print('nop')
        return {'team_id': self.team.pk, 'id': self.id, 'after': (now() - timedelta(days=90)).strftime('%Y-%m-%dT%H:%M:%S.%f'), 'before': now().strftime('%Y-%m-%dT%H:%M:%S.%f')}