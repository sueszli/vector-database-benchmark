from typing import Any, Dict, List, Optional, Tuple, Union, cast
from posthog.clickhouse.materialized_columns import ColumnName
from posthog.constants import PropertyOperatorType
from posthog.models import Filter, Team
from posthog.models.action import Action
from posthog.models.cohort import Cohort
from posthog.models.cohort.util import format_static_cohort_query, get_count_operator, get_entity_query
from posthog.models.filters.mixins.utils import cached_property
from posthog.models.property import BehavioralPropertyType, OperatorInterval, Property, PropertyGroup, PropertyName
from posthog.models.property.util import prop_filter_json_extract
from posthog.queries.event_query import EventQuery
from posthog.queries.util import PersonPropertiesMode
from posthog.utils import PersonOnEventsMode
Relative_Date = Tuple[int, OperatorInterval]
Event = Tuple[str, Union[str, int]]
INTERVAL_TO_SECONDS = {'minute': 60, 'hour': 3600, 'day': 86400, 'week': 604800, 'month': 2592000, 'year': 31536000}

def relative_date_to_seconds(date: Tuple[Optional[int], Union[OperatorInterval, None]]):
    if False:
        i = 10
        return i + 15
    if date[0] is None or date[1] is None:
        raise ValueError('Time value and time interval must be specified')
    return date[0] * INTERVAL_TO_SECONDS[date[1]]

def validate_interval(interval: Optional[OperatorInterval]) -> OperatorInterval:
    if False:
        for i in range(10):
            print('nop')
    if interval is None or interval not in INTERVAL_TO_SECONDS.keys():
        raise ValueError(f'Invalid interval: {interval}')
    else:
        return interval

def parse_and_validate_positive_integer(value: Optional[int], value_name: str) -> int:
    if False:
        return 10
    if value is None:
        raise ValueError(f'{value_name} cannot be None')
    try:
        parsed_value = int(value)
    except ValueError:
        raise ValueError(f'{value_name} must be an integer, got {value}')
    if parsed_value <= 0:
        raise ValueError(f'{value_name} must be greater than 0, got {value}')
    return parsed_value

def validate_entity(possible_event: Tuple[Optional[str], Optional[Union[int, str]]]) -> Event:
    if False:
        for i in range(10):
            print('nop')
    event_type = possible_event[0]
    event_val = possible_event[1]
    if event_type is None or event_val is None:
        raise ValueError('Entity name and entity id must be specified')
    return (event_type, event_val)

def validate_seq_date_more_recent_than_date(seq_date: Relative_Date, date: Relative_Date):
    if False:
        print('Hello World!')
    if relative_date_is_greater(seq_date, date):
        raise ValueError('seq_date must be more recent than date')

def relative_date_is_greater(date_1: Relative_Date, date_2: Relative_Date) -> bool:
    if False:
        while True:
            i = 10
    return relative_date_to_seconds(date_1) > relative_date_to_seconds(date_2)

def convert_to_entity_params(events: List[Event]) -> Tuple[List, List]:
    if False:
        i = 10
        return i + 15
    res_events = []
    res_actions = []
    for (idx, event) in enumerate(events):
        event_type = event[0]
        event_val = event[1]
        if event_type == 'events':
            res_events.append({'id': event_val, 'name': event_val, 'order': idx, 'type': event_type})
        elif event_type == 'actions':
            action = Action.objects.get(id=event_val)
            res_actions.append({'id': event_val, 'name': action.name, 'order': idx, 'type': event_type})
    return (res_events, res_actions)

def get_relative_date_arg(relative_date: Relative_Date) -> str:
    if False:
        i = 10
        return i + 15
    return f'-{relative_date[0]}{relative_date[1][0].lower()}'

def full_outer_join_query(q: str, alias: str, left_operand: str, right_operand: str) -> str:
    if False:
        i = 10
        return i + 15
    return join_query(q, 'FULL OUTER JOIN', alias, left_operand, right_operand)

def inner_join_query(q: str, alias: str, left_operand: str, right_operand: str) -> str:
    if False:
        print('Hello World!')
    return join_query(q, 'INNER JOIN', alias, left_operand, right_operand)

def join_query(q: str, join: str, alias: str, left_operand: str, right_operand: str) -> str:
    if False:
        print('Hello World!')
    return f'{join} ({q}) {alias} ON {left_operand} = {right_operand}'

def if_condition(condition: str, true_res: str, false_res: str) -> str:
    if False:
        return 10
    return f'if({condition}, {true_res}, {false_res})'

class FOSSCohortQuery(EventQuery):
    BEHAVIOR_QUERY_ALIAS = 'behavior_query'
    FUNNEL_QUERY_ALIAS = 'funnel_query'
    SEQUENCE_FIELD_ALIAS = 'steps'
    _fields: List[str]
    _events: List[str]
    _earliest_time_for_event_query: Optional[Relative_Date]
    _restrict_event_query_by_time: bool

    def __init__(self, filter: Filter, team: Team, *, cohort_pk: Optional[int]=None, round_interval=False, should_join_distinct_ids=False, should_join_persons=False, extra_fields: List[ColumnName]=[], extra_event_properties: List[PropertyName]=[], extra_person_fields: List[ColumnName]=[], override_aggregate_users_by_distinct_id: Optional[bool]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self._fields = []
        self._events = []
        self._earliest_time_for_event_query = None
        self._restrict_event_query_by_time = True
        self._cohort_pk = cohort_pk
        super().__init__(filter=FOSSCohortQuery.unwrap_cohort(filter, team.pk), team=team, round_interval=round_interval, should_join_distinct_ids=should_join_distinct_ids, should_join_persons=should_join_persons, extra_fields=extra_fields, extra_event_properties=extra_event_properties, extra_person_fields=extra_person_fields, override_aggregate_users_by_distinct_id=override_aggregate_users_by_distinct_id, person_on_events_mode=team.person_on_events_mode, **kwargs)
        self._validate_negations()
        property_groups = self._column_optimizer.property_optimizer.parse_property_groups(self._filter.property_groups)
        self._inner_property_groups = property_groups.inner
        self._outer_property_groups = property_groups.outer

    @staticmethod
    def unwrap_cohort(filter: Filter, team_id: int) -> Filter:
        if False:
            return 10

        def _unwrap(property_group: PropertyGroup, negate_group: bool=False) -> PropertyGroup:
            if False:
                print('Hello World!')
            if len(property_group.values):
                if isinstance(property_group.values[0], PropertyGroup):
                    if not negate_group:
                        return PropertyGroup(type=property_group.type, values=[_unwrap(v) for v in cast(List[PropertyGroup], property_group.values)])
                    else:
                        return PropertyGroup(type=PropertyOperatorType.AND if property_group.type == PropertyOperatorType.OR else PropertyOperatorType.OR, values=[_unwrap(v, True) for v in cast(List[PropertyGroup], property_group.values)])
                elif isinstance(property_group.values[0], Property):
                    new_property_group_list: List[PropertyGroup] = []
                    for prop in property_group.values:
                        prop = cast(Property, prop)
                        current_negation = prop.negation or False
                        negation_value = not current_negation if negate_group else current_negation
                        if prop.type in ['cohort', 'precalculated-cohort']:
                            try:
                                prop_cohort: Cohort = Cohort.objects.get(pk=prop.value, team_id=team_id)
                                if prop_cohort.is_static:
                                    new_property_group_list.append(PropertyGroup(type=PropertyOperatorType.AND, values=[Property(type='static-cohort', key='id', value=prop_cohort.pk, negation=negation_value)]))
                                else:
                                    new_property_group_list.append(_unwrap(prop_cohort.properties, negation_value))
                            except Cohort.DoesNotExist:
                                new_property_group_list.append(PropertyGroup(type=PropertyOperatorType.AND, values=[Property(key='fake_key_01r2ho', value=0, type='person')]))
                        else:
                            prop.negation = negation_value
                            new_property_group_list.append(PropertyGroup(type=PropertyOperatorType.AND, values=[prop]))
                    if not negate_group:
                        return PropertyGroup(type=property_group.type, values=new_property_group_list)
                    else:
                        return PropertyGroup(type=PropertyOperatorType.AND if property_group.type == PropertyOperatorType.OR else PropertyOperatorType.OR, values=new_property_group_list)
            return property_group
        new_props = _unwrap(filter.property_groups)
        return filter.shallow_clone({'properties': new_props.to_dict()})

    def get_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        if not self._outer_property_groups:
            return self._person_query.get_query(prepend=self._cohort_pk)
        (conditions, condition_params) = self._get_conditions()
        self.params.update(condition_params)
        subq = []
        (behavior_subquery, behavior_subquery_params, behavior_query_alias) = self._get_behavior_subquery()
        subq.append((behavior_subquery, behavior_query_alias))
        self.params.update(behavior_subquery_params)
        (person_query, person_params, person_query_alias) = self._get_persons_query(prepend=str(self._cohort_pk))
        subq.append((person_query, person_query_alias))
        self.params.update(person_params)
        (q, fields) = self._build_sources(subq)
        final_query = f'\n        SELECT {fields} AS id  FROM\n        {q}\n        WHERE 1 = 1\n        {conditions}\n        '
        return (final_query, self.params)

    def _build_sources(self, subq: List[Tuple[str, str]]) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        q = ''
        filtered_queries = [(q, alias) for (q, alias) in subq if q and len(q)]
        prev_alias: Optional[str] = None
        fields = ''
        for (idx, (subq_query, subq_alias)) in enumerate(filtered_queries):
            if idx == 0:
                q += f'({subq_query}) {subq_alias}'
                fields = f'{subq_alias}.person_id'
            elif prev_alias:
                if subq_alias == self.PERSON_TABLE_ALIAS and self.should_pushdown_persons:
                    if self._person_on_events_mode == PersonOnEventsMode.V1_ENABLED:
                        continue
                    q = f"{q} {inner_join_query(subq_query, subq_alias, f'{subq_alias}.person_id', f'{prev_alias}.person_id')}"
                    fields = f'{subq_alias}.person_id'
                else:
                    q = f"{q} {full_outer_join_query(subq_query, subq_alias, f'{subq_alias}.person_id', f'{prev_alias}.person_id')}"
                    fields = if_condition(f"{prev_alias}.person_id = '00000000-0000-0000-0000-000000000000'", f'{subq_alias}.person_id', f'{fields}')
            prev_alias = subq_alias
        return (q, fields)

    def _get_behavior_subquery(self) -> Tuple[str, Dict[str, Any], str]:
        if False:
            print('Hello World!')
        event_param_name = f'{self._cohort_pk}_event_ids'
        person_prop_query = ''
        person_prop_params: dict = {}
        (query, params) = ('', {})
        if self._should_join_behavioral_query:
            _fields = [f'{(self.DISTINCT_ID_TABLE_ALIAS if self._person_on_events_mode == PersonOnEventsMode.DISABLED else self.EVENT_TABLE_ALIAS)}.person_id AS person_id']
            _fields.extend(self._fields)
            if self.should_pushdown_persons and self._person_on_events_mode != PersonOnEventsMode.DISABLED:
                (person_prop_query, person_prop_params) = self._get_prop_groups(self._inner_property_groups, person_properties_mode=PersonPropertiesMode.DIRECT_ON_EVENTS, person_id_joined_alias=self._person_id_alias)
            (date_condition, date_params) = self._get_date_condition()
            query = f"\n            SELECT {', '.join(_fields)} FROM events {self.EVENT_TABLE_ALIAS}\n            {self._get_person_ids_query()}\n            WHERE team_id = %(team_id)s\n            AND event IN %({event_param_name})s\n            {date_condition}\n            {person_prop_query}\n            GROUP BY person_id\n            "
            (query, params) = (query, {'team_id': self._team_id, event_param_name: self._events, **date_params, **person_prop_params})
        return (query, params, self.BEHAVIOR_QUERY_ALIAS)

    def _get_persons_query(self, prepend: str='') -> Tuple[str, Dict[str, Any], str]:
        if False:
            print('Hello World!')
        (query, params) = ('', {})
        if self._should_join_persons:
            (person_query, person_params) = self._person_query.get_query(prepend=prepend)
            person_query = f'SELECT *, id AS person_id FROM ({person_query})'
            (query, params) = (person_query, person_params)
        return (query, params, self.PERSON_TABLE_ALIAS)

    @cached_property
    def should_pushdown_persons(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return 'person' not in [prop.type for prop in getattr(self._outer_property_groups, 'flat', [])] and 'static-cohort' not in [prop.type for prop in getattr(self._outer_property_groups, 'flat', [])]

    def _get_date_condition(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        date_query = ''
        date_params: Dict[str, Any] = {}
        earliest_time_param = f'earliest_time_{self._cohort_pk}'
        if self._earliest_time_for_event_query and self._restrict_event_query_by_time:
            date_params = {earliest_time_param: self._earliest_time_for_event_query[0]}
            date_query = f'AND timestamp <= now() AND timestamp >= now() - INTERVAL %({earliest_time_param})s {self._earliest_time_for_event_query[1]}'
        return (date_query, date_params)

    def _check_earliest_date(self, relative_date: Relative_Date) -> None:
        if False:
            return 10
        if self._earliest_time_for_event_query is None:
            self._earliest_time_for_event_query = relative_date
        elif relative_date_is_greater(relative_date, self._earliest_time_for_event_query):
            self._earliest_time_for_event_query = relative_date

    def _get_conditions(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10

        def build_conditions(prop: Optional[Union[PropertyGroup, Property]], prepend='level', num=0):
            if False:
                print('Hello World!')
            if not prop:
                return ('', {})
            if isinstance(prop, PropertyGroup):
                params = {}
                conditions = []
                for (idx, p) in enumerate(prop.values):
                    (q, q_params) = build_conditions(p, f'{prepend}_level_{num}', idx)
                    if q != '':
                        conditions.append(q)
                        params.update(q_params)
                return (f"({f' {prop.type} '.join(conditions)})", params)
            else:
                return self._get_condition_for_property(prop, prepend, num)
        (conditions, params) = build_conditions(self._outer_property_groups, prepend=f'{self._cohort_pk}_level', num=0)
        return (f'AND ({conditions})' if conditions else '', params)

    def _get_condition_for_property(self, prop: Property, prepend: str, idx: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        res: str = ''
        params: Dict[str, Any] = {}
        if prop.type == 'behavioral':
            if prop.value == 'performed_event':
                (res, params) = self.get_performed_event_condition(prop, prepend, idx)
            elif prop.value == 'performed_event_multiple':
                (res, params) = self.get_performed_event_multiple(prop, prepend, idx)
        elif prop.type == 'person':
            (res, params) = self.get_person_condition(prop, prepend, idx)
        elif prop.type == 'static-cohort':
            (res, params) = self.get_static_cohort_condition(prop, prepend, idx)
        else:
            raise ValueError(f'Invalid property type for Cohort queries: {prop.type}')
        return (res, params)

    def get_person_condition(self, prop: Property, prepend: str, idx: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        if self._outer_property_groups and len(self._outer_property_groups.flat):
            return prop_filter_json_extract(prop, idx, prepend, prop_var='person_props', allow_denormalized_props=True, property_operator='')
        else:
            return ('', {})

    def get_static_cohort_condition(self, prop: Property, prepend: str, idx: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        cohort = Cohort.objects.get(pk=cast(int, prop.value))
        (query, params) = format_static_cohort_query(cohort, idx, prepend)
        return (f"id {('NOT' if prop.negation else '')} IN ({query})", params)

    def get_performed_event_condition(self, prop: Property, prepend: str, idx: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        event = (prop.event_type, prop.key)
        column_name = f'performed_event_condition_{prepend}_{idx}'
        (entity_query, entity_params) = self._get_entity(event, prepend, idx)
        date_value = parse_and_validate_positive_integer(prop.time_value, 'time_value')
        date_interval = validate_interval(prop.time_interval)
        date_param = f'{prepend}_date_{idx}'
        self._check_earliest_date((date_value, date_interval))
        field = f'countIf(timestamp > now() - INTERVAL %({date_param})s {date_interval} AND timestamp < now() AND {entity_query}) > 0 AS {column_name}'
        self._fields.append(field)
        return (f"{('NOT' if prop.negation else '')} {column_name}", {f'{date_param}': date_value, **entity_params})

    def get_performed_event_multiple(self, prop: Property, prepend: str, idx: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        event = (prop.event_type, prop.key)
        column_name = f'performed_event_multiple_condition_{prepend}_{idx}'
        (entity_query, entity_params) = self._get_entity(event, prepend, idx)
        count = parse_and_validate_positive_integer(prop.operator_value, 'operator_value')
        date_value = parse_and_validate_positive_integer(prop.time_value, 'time_value')
        date_interval = validate_interval(prop.time_interval)
        date_param = f'{prepend}_date_{idx}'
        operator_value_param = f'{prepend}_operator_value_{idx}'
        self._check_earliest_date((date_value, date_interval))
        field = f'countIf(timestamp > now() - INTERVAL %({date_param})s {date_interval} AND timestamp < now() AND {entity_query}) {get_count_operator(prop.operator)} %({operator_value_param})s AS {column_name}'
        self._fields.append(field)
        return (f"{('NOT' if prop.negation else '')} {column_name}", {f'{operator_value_param}': count, f'{date_param}': date_value, **entity_params})

    def _determine_should_join_distinct_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._should_join_distinct_ids = self._person_on_events_mode != PersonOnEventsMode.V1_ENABLED

    def _determine_should_join_persons(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._should_join_persons = self._column_optimizer.is_using_person_properties or len(self._column_optimizer.used_properties_with_type('static-cohort')) > 0

    @cached_property
    def _should_join_behavioral_query(self) -> bool:
        if False:
            while True:
                i = 10
        for prop in self._filter.property_groups.flat:
            if prop.value in [BehavioralPropertyType.PERFORMED_EVENT, BehavioralPropertyType.PERFORMED_EVENT_FIRST_TIME, BehavioralPropertyType.PERFORMED_EVENT_MULTIPLE, BehavioralPropertyType.PERFORMED_EVENT_REGULARLY, BehavioralPropertyType.RESTARTED_PERFORMING_EVENT, BehavioralPropertyType.STOPPED_PERFORMING_EVENT]:
                return True
        return False

    def _validate_negations(self) -> None:
        if False:
            print('Hello World!')
        pass

    def _get_entity(self, event: Tuple[Optional[str], Optional[Union[int, str]]], prepend: str, idx: int) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        res: str = ''
        params: Dict[str, Any] = {}
        if event[0] is None or event[1] is None:
            raise ValueError('Event type and key must be specified')
        if event[0] == 'actions':
            self._add_action(int(event[1]))
            (res, params) = get_entity_query(None, int(event[1]), self._team_id, f'{prepend}_entity_{idx}', self._filter.hogql_context)
        elif event[0] == 'events':
            self._add_event(str(event[1]))
            (res, params) = get_entity_query(str(event[1]), None, self._team_id, f'{prepend}_entity_{idx}', self._filter.hogql_context)
        else:
            raise ValueError(f"Event type must be 'events' or 'actions'")
        return (res, params)

    def _add_action(self, action_id: int) -> None:
        if False:
            print('Hello World!')
        action = Action.objects.get(id=action_id)
        for step in action.steps.all():
            self._events.append(step.event)

    def _add_event(self, event_id: str) -> None:
        if False:
            return 10
        self._events.append(event_id)