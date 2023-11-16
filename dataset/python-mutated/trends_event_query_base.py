from typing import Any, Dict, Tuple
from posthog.constants import MONTHLY_ACTIVE, UNIQUE_USERS, WEEKLY_ACTIVE, PropertyOperatorType
from posthog.models import Entity
from posthog.models.entity.util import get_entity_filtering_params
from posthog.models.filters.filter import Filter
from posthog.models.filters.mixins.utils import cached_property
from posthog.queries.event_query import EventQuery
from posthog.queries.person_query import PersonQuery
from posthog.queries.query_date_range import QueryDateRange
from posthog.queries.trends.util import COUNT_PER_ACTOR_MATH_FUNCTIONS, get_active_user_params
from posthog.queries.util import get_person_properties_mode
from posthog.utils import PersonOnEventsMode

class TrendsEventQueryBase(EventQuery):
    _entity: Entity
    _filter: Filter

    def __init__(self, entity: Entity, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._entity = entity
        super().__init__(*args, **kwargs)

    def get_query_base(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        '\n        Returns part of the event query with only FROM, JOINs and WHERE clauses.\n        '
        (date_query, date_params) = self._get_date_filter()
        self.params.update(date_params)
        (prop_query, prop_params) = self._get_prop_groups(self._filter.property_groups.combine_property_group(PropertyOperatorType.AND, self._entity.property_groups), person_properties_mode=get_person_properties_mode(self._team), person_id_joined_alias=self._person_id_alias)
        self.params.update(prop_params)
        (entity_query, entity_params) = self._get_entity_query()
        self.params.update(entity_params)
        (person_query, person_params) = self._get_person_query()
        self.params.update(person_params)
        (groups_query, groups_params) = self._get_groups_query()
        self.params.update(groups_params)
        (session_query, session_params) = self._get_sessions_query()
        self.params.update(session_params)
        sample_clause = 'SAMPLE %(sampling_factor)s' if self._filter.sampling_factor else ''
        self.params.update({'sampling_factor': self._filter.sampling_factor})
        query = f'\n            FROM events {self.EVENT_TABLE_ALIAS}\n            {sample_clause}\n            {self._get_person_ids_query()}\n            {person_query}\n            {groups_query}\n            {session_query}\n            WHERE team_id = %(team_id)s\n            {entity_query}\n            {date_query}\n            {prop_query}\n            {self._get_not_null_actor_condition()}\n        '
        return (query, self.params)

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._person_on_events_mode == PersonOnEventsMode.V1_ENABLED:
            self._should_join_distinct_ids = False
        is_entity_per_user = self._entity.math in (UNIQUE_USERS, WEEKLY_ACTIVE, MONTHLY_ACTIVE, *COUNT_PER_ACTOR_MATH_FUNCTIONS.keys())
        if is_entity_per_user and (not self._aggregate_users_by_distinct_id) and (self._entity.math_group_type_index is None) or self._column_optimizer.is_using_cohort_propertes:
            self._should_join_distinct_ids = True

    def _determine_should_join_persons(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._person_on_events_mode != PersonOnEventsMode.DISABLED:
            self._should_join_persons = False
        else:
            EventQuery._determine_should_join_persons(self)

    def _get_not_null_actor_condition(self) -> str:
        if False:
            return 10
        if self._entity.math_group_type_index is None:
            return f'AND notEmpty({self.EVENT_TABLE_ALIAS}.person_id)' if self._person_on_events_mode != PersonOnEventsMode.DISABLED else ''
        else:
            return f"""AND "$group_{self._entity.math_group_type_index}" != ''"""

    def _get_date_filter(self) -> Tuple[str, Dict]:
        if False:
            for i in range(10):
                print('nop')
        date_query = ''
        date_params: Dict[str, Any] = {}
        query_date_range = QueryDateRange(self._filter, self._team)
        (parsed_date_from, date_from_params) = query_date_range.date_from
        (parsed_date_to, date_to_params) = query_date_range.date_to
        date_params.update(date_from_params)
        date_params.update(date_to_params)
        self.parsed_date_from = parsed_date_from
        self.parsed_date_to = parsed_date_to
        if self._entity.math in [WEEKLY_ACTIVE, MONTHLY_ACTIVE]:
            (active_user_format_params, active_user_query_params) = get_active_user_params(self._filter, self._entity, self._team_id)
            self.active_user_params = active_user_format_params
            date_params.update(active_user_query_params)
            date_query = '{parsed_date_from_prev_range} {parsed_date_to}'.format(**active_user_format_params, parsed_date_to=parsed_date_to)
        else:
            date_query = '{parsed_date_from} {parsed_date_to}'.format(parsed_date_from=parsed_date_from, parsed_date_to=parsed_date_to)
        return (date_query, date_params)

    def _get_entity_query(self) -> Tuple[str, Dict]:
        if False:
            return 10
        (entity_params, entity_format_params) = get_entity_filtering_params(allowed_entities=[self._entity], team_id=self._team_id, table_name=self.EVENT_TABLE_ALIAS, person_properties_mode=get_person_properties_mode(self._team), hogql_context=self._filter.hogql_context, person_id_joined_alias=self._person_id_alias)
        return (entity_format_params['entity_query'], entity_params)

    @cached_property
    def _person_query(self):
        if False:
            print('Hello World!')
        return PersonQuery(self._filter, self._team_id, self._column_optimizer, extra_fields=self._extra_person_fields, entity=self._entity)