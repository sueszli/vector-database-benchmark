from typing import Any, Dict, Set, Tuple, Union
from posthog.constants import TREND_FILTER_TYPE_ACTIONS
from posthog.hogql.hogql import translate_hogql
from posthog.models.filters.filter import Filter
from posthog.models.group.util import get_aggregation_target_field
from posthog.queries.event_query import EventQuery
from posthog.queries.util import get_person_properties_mode
from posthog.utils import PersonOnEventsMode

class FunnelEventQuery(EventQuery):
    _filter: Filter

    def get_query(self, entities=None, entity_name='events', skip_entity_filter=False) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        if self._filter.aggregation_group_type_index is not None:
            aggregation_target = get_aggregation_target_field(self._filter.aggregation_group_type_index, self.EVENT_TABLE_ALIAS, self._person_id_alias)
        elif self._filter.funnel_aggregate_by_hogql and self._filter.funnel_aggregate_by_hogql != 'person_id':
            aggregation_target = translate_hogql(self._filter.funnel_aggregate_by_hogql, events_table_alias=self.EVENT_TABLE_ALIAS, context=self._filter.hogql_context)
        elif self._aggregate_users_by_distinct_id:
            aggregation_target = f'{self.EVENT_TABLE_ALIAS}.distinct_id'
        else:
            aggregation_target = self._person_id_alias
        _fields = [f'{self.EVENT_TABLE_ALIAS}.timestamp as timestamp', f'{aggregation_target} as aggregation_target']
        _fields += [f'{self.EVENT_TABLE_ALIAS}.{field} AS {field}' for field in self._extra_fields]
        if self._person_on_events_mode != PersonOnEventsMode.DISABLED:
            _fields += [f'{self._person_id_alias} as person_id']
            _fields.extend((f'{self.EVENT_TABLE_ALIAS}."{column_name}" as "{column_name}"' for column_name in sorted(self._column_optimizer.person_on_event_columns_to_query)))
        else:
            if self._should_join_distinct_ids:
                _fields += [f'{self._person_id_alias} as person_id']
            if self._should_join_persons:
                _fields.extend((f'{self.PERSON_TABLE_ALIAS}.{column_name} as {column_name}' for column_name in sorted(self._person_query.fields)))
        _fields = list(filter(None, _fields))
        (date_query, date_params) = self._get_date_filter()
        self.params.update(date_params)
        (prop_query, prop_params) = self._get_prop_groups(self._filter.property_groups, person_properties_mode=get_person_properties_mode(self._team), person_id_joined_alias=self._person_id_alias)
        self.params.update(prop_params)
        if skip_entity_filter:
            entity_query = ''
            entity_params: Dict[str, Any] = {}
        else:
            (entity_query, entity_params) = self._get_entity_query(entities, entity_name)
        self.params.update(entity_params)
        (person_query, person_params) = self._get_person_query()
        self.params.update(person_params)
        (groups_query, groups_params) = self._get_groups_query()
        self.params.update(groups_params)
        null_person_filter = f'AND notEmpty({self.EVENT_TABLE_ALIAS}.person_id)' if self._person_on_events_mode != PersonOnEventsMode.DISABLED else ''
        sample_clause = 'SAMPLE %(sampling_factor)s' if self._filter.sampling_factor else ''
        self.params.update({'sampling_factor': self._filter.sampling_factor})
        query = f"\n            SELECT {', '.join(_fields)}\n            {{extra_select_fields}}\n            FROM events {self.EVENT_TABLE_ALIAS}\n            {sample_clause}\n            {self._get_person_ids_query()}\n            {person_query}\n            {groups_query}\n            {{extra_join}}\n            WHERE team_id = %(team_id)s\n            {entity_query}\n            {date_query}\n            {prop_query}\n            {null_person_filter}\n            {{step_filter}}\n        "
        return (query, self.params)

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            while True:
                i = 10
        non_person_id_aggregation = self._filter.aggregation_group_type_index is not None or self._aggregate_users_by_distinct_id
        is_using_cohort_propertes = self._column_optimizer.is_using_cohort_propertes
        if self._person_on_events_mode == PersonOnEventsMode.V2_ENABLED:
            self._should_join_distinct_ids = True
        elif self._person_on_events_mode == PersonOnEventsMode.V1_ENABLED or (non_person_id_aggregation and (not is_using_cohort_propertes)):
            self._should_join_distinct_ids = False
        else:
            self._should_join_distinct_ids = True

    def _determine_should_join_persons(self) -> None:
        if False:
            i = 10
            return i + 15
        EventQuery._determine_should_join_persons(self)
        if self._person_on_events_mode != PersonOnEventsMode.DISABLED:
            self._should_join_persons = False

    def _get_entity_query(self, entities=None, entity_name='events') -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        events: Set[Union[int, str, None]] = set()
        entities_to_use = entities or self._filter.entities
        for entity in entities_to_use:
            if entity.type == TREND_FILTER_TYPE_ACTIONS:
                action = entity.get_action()
                events.update(action.get_step_events())
            else:
                events.add(entity.id)
        if None in events:
            return ('AND 1 = 1', {})
        return (f'AND event IN %({entity_name})s', {entity_name: sorted(list(events))})