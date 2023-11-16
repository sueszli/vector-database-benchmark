from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID
from posthog.clickhouse.materialized_columns import ColumnName
from posthog.constants import PropertyOperatorType
from posthog.models import Filter
from posthog.models.cohort import Cohort
from posthog.models.cohort.sql import GET_COHORTPEOPLE_BY_COHORT_ID, GET_STATIC_COHORTPEOPLE_BY_COHORT_ID
from posthog.models.cohort.util import format_precalculated_cohort_query, format_static_cohort_query
from posthog.models.entity import Entity
from posthog.models.filters.path_filter import PathFilter
from posthog.models.filters.retention_filter import RetentionFilter
from posthog.models.filters.stickiness_filter import StickinessFilter
from posthog.models.property import Property, PropertyGroup
from posthog.models.property.util import extract_tables_and_properties, parse_prop_grouped_clauses, prop_filter_json_extract
from posthog.queries.column_optimizer.column_optimizer import ColumnOptimizer
from posthog.queries.person_distinct_id_query import get_team_distinct_ids_query
from posthog.queries.trends.util import COUNT_PER_ACTOR_MATH_FUNCTIONS
from posthog.queries.util import PersonPropertiesMode

class PersonQuery:
    """
    Query class responsible for joining with `person` clickhouse table

    For sake of performance, this class:
    - Tries to do as much person property filtering as possible here
    - Minimizes the amount of columns read
    """
    PERSON_PROPERTIES_ALIAS = 'person_props'
    COHORT_TABLE_ALIAS = 'cohort_persons'
    ALIASES = {'properties': 'person_props'}
    _filter: Union[Filter, PathFilter, RetentionFilter, StickinessFilter]
    _team_id: int
    _column_optimizer: ColumnOptimizer
    _extra_fields: Set[ColumnName]
    _inner_person_properties: Optional[PropertyGroup]
    _cohort: Optional[Cohort]
    _include_distinct_ids: Optional[bool] = False

    def __init__(self, filter: Union[Filter, PathFilter, RetentionFilter, StickinessFilter], team_id: int, column_optimizer: Optional[ColumnOptimizer]=None, cohort: Optional[Cohort]=None, *, entity: Optional[Entity]=None, extra_fields: Optional[List[ColumnName]]=None, cohort_filters: Optional[List[Property]]=None, include_distinct_ids: Optional[bool]=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._filter = filter
        self._team_id = team_id
        self._entity = entity
        self._cohort = cohort
        self._column_optimizer = column_optimizer or ColumnOptimizer(self._filter, self._team_id)
        self._extra_fields = set(extra_fields) if extra_fields else set()
        self._cohort_filters = cohort_filters
        self._include_distinct_ids = include_distinct_ids
        if self.PERSON_PROPERTIES_ALIAS in self._extra_fields:
            self._extra_fields = self._extra_fields - {self.PERSON_PROPERTIES_ALIAS} | {'properties'}
        properties = self._filter.property_groups.combine_property_group(PropertyOperatorType.AND, self._entity.property_groups if self._entity else None)
        self._inner_person_properties = self._column_optimizer.property_optimizer.parse_property_groups(properties).inner

    def get_query(self, prepend: Optional[Union[str, int]]=None, paginate: bool=False, filter_future_persons: bool=False) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        prepend = str(prepend) if prepend is not None else ''
        fields = 'id' + ' '.join((f', argMax({column_name}, version) as {alias}' for (column_name, alias) in self._get_fields()))
        (person_filters_prefiltering_condition, person_filters_finalization_condition, person_filters_params) = self._get_person_filter_clauses(prepend=prepend)
        (multiple_cohorts_condition, multiple_cohorts_params) = self._get_multiple_cohorts_clause(prepend=prepend)
        (single_cohort_join, single_cohort_params) = self._get_fast_single_cohort_clause()
        if paginate:
            order = 'ORDER BY argMax(person.created_at, version) DESC, id DESC' if paginate else ''
            (limit_offset, limit_params) = self._get_limit_offset_clause()
        else:
            order = ''
            (limit_offset, limit_params) = ('', {})
        (search_prefiltering_condition, search_finalization_condition, search_params) = self._get_search_clauses(prepend=prepend)
        (distinct_id_condition, distinct_id_params) = self._get_distinct_id_clause()
        (email_condition, email_params) = self._get_email_clause()
        filter_future_persons_condition = 'AND argMax(person.created_at, version) < now() + INTERVAL 1 DAY' if filter_future_persons else ''
        (updated_after_condition, updated_after_params) = self._get_updated_after_clause()
        prefiltering_lookup = f'AND id IN (\n            SELECT id FROM person\n            {single_cohort_join}\n            WHERE team_id = %(team_id)s\n            {person_filters_prefiltering_condition}\n            {search_prefiltering_condition}\n        )\n        ' if person_filters_prefiltering_condition or search_prefiltering_condition else ''
        top_level_single_cohort_join = single_cohort_join if not prefiltering_lookup else ''
        return self._add_distinct_id_join_if_needed(f'\n            SELECT {fields}\n            FROM person\n            {top_level_single_cohort_join}\n            WHERE team_id = %(team_id)s\n            {prefiltering_lookup}\n            {multiple_cohorts_condition}\n            GROUP BY id\n            HAVING max(is_deleted) = 0\n            {filter_future_persons_condition} {updated_after_condition}\n            {person_filters_finalization_condition} {search_finalization_condition}\n            {distinct_id_condition} {email_condition}\n            {order}\n            {limit_offset}\n            SETTINGS optimize_aggregation_in_order = 1\n            ', {**updated_after_params, **person_filters_params, **single_cohort_params, **limit_params, **search_params, **distinct_id_params, **email_params, **multiple_cohorts_params, 'team_id': self._team_id})

    @property
    def fields(self) -> List[ColumnName]:
        if False:
            return 10
        'Returns person table fields this query exposes'
        return [alias for (column_name, alias) in self._get_fields()]

    @property
    def is_used(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether properties or any other columns are actually being queried'
        if any((self._uses_person_id(prop) for prop in self._filter.property_groups.flat)):
            return True
        for entity in self._filter.entities:
            is_count_per_user = entity.math in COUNT_PER_ACTOR_MATH_FUNCTIONS and entity.math_group_type_index is None
            if is_count_per_user or any((self._uses_person_id(prop) for prop in entity.property_groups.flat)):
                return True
        return len(self._column_optimizer.person_columns_to_query) > 0

    def _uses_person_id(self, prop: Property) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return prop.type in ('person', 'static-cohort', 'precalculated-cohort')

    def _get_fields(self) -> List[Tuple[str, str]]:
        if False:
            i = 10
            return i + 15
        properties_to_query = self._column_optimizer.used_properties_with_type('person')
        if self._inner_person_properties:
            properties_to_query -= extract_tables_and_properties(self._inner_person_properties.flat)
        columns = self._column_optimizer.columns_to_query('person', set(properties_to_query)) | set(self._extra_fields)
        return [(column_name, self.ALIASES.get(column_name, column_name)) for column_name in sorted(columns)]

    def _get_person_filter_clauses(self, prepend: str='') -> Tuple[str, str, Dict]:
        if False:
            print('Hello World!')
        (finalization_conditions, params) = parse_prop_grouped_clauses(self._team_id, self._inner_person_properties, has_person_id_joined=False, group_properties_joined=False, person_properties_mode=PersonPropertiesMode.DIRECT, prepend=f'person_filter_fin_{prepend}', hogql_context=self._filter.hogql_context)
        (prefiltering_conditions, prefiltering_params) = parse_prop_grouped_clauses(self._team_id, self._inner_person_properties, has_person_id_joined=False, group_properties_joined=False, person_properties_mode=PersonPropertiesMode.DIRECT_ON_PERSONS, prepend=f'person_filter_pre_{prepend}', hogql_context=self._filter.hogql_context)
        params.update(prefiltering_params)
        return (prefiltering_conditions, finalization_conditions, params)

    def _get_fast_single_cohort_clause(self) -> Tuple[str, Dict]:
        if False:
            return 10
        if self._cohort:
            cohort_table = GET_STATIC_COHORTPEOPLE_BY_COHORT_ID if self._cohort.is_static else GET_COHORTPEOPLE_BY_COHORT_ID
            return (f'\n            INNER JOIN (\n                {cohort_table}\n            ) {self.COHORT_TABLE_ALIAS}\n            ON {self.COHORT_TABLE_ALIAS}.person_id = person.id\n            ', {'team_id': self._team_id, 'cohort_id': self._cohort.pk, 'version': self._cohort.version})
        else:
            return ('', {})

    def _get_multiple_cohorts_clause(self, prepend: str='') -> Tuple[str, Dict]:
        if False:
            while True:
                i = 10
        if self._cohort_filters:
            query = []
            params: Dict[str, Any] = {}
            for (index, property) in enumerate(self._cohort_filters):
                try:
                    cohort = Cohort.objects.get(pk=property.value, team_id=self._team_id)
                    if property.type == 'static-cohort':
                        (subquery, subquery_params) = format_static_cohort_query(cohort, index, prepend)
                    else:
                        (subquery, subquery_params) = format_precalculated_cohort_query(cohort, index, prepend)
                    query.append(f'AND id in ({subquery})')
                    params.update(**subquery_params)
                except Cohort.DoesNotExist:
                    continue
            return (' '.join(query), params)
        else:
            return ('', {})

    def _get_limit_offset_clause(self) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        if not isinstance(self._filter, Filter):
            return ('', {})
        if not (self._filter.limit or self._filter.offset):
            return ('', {})
        clause = ''
        params = {}
        if self._filter.limit:
            clause += ' LIMIT %(limit)s'
            params.update({'limit': self._filter.limit})
        if self._filter.offset:
            clause += ' OFFSET %(offset)s'
            params.update({'offset': self._filter.offset})
        return (clause, params)

    def _get_search_clauses(self, prepend: str='') -> Tuple[str, str, Dict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return - respectively - the prefiltering search clause (not aggregated by is_deleted or version, which is great\n        for memory usage), the final search clause (aggregated for true results, more expensive), and new params.\n        '
        if not isinstance(self._filter, Filter):
            return ('', '', {})
        if self._filter.search:
            id_conditions_param = f'id_conditions_{prepend}'
            id_conditions_sql = f'\n            id IN (\n                SELECT person_id FROM ({get_team_distinct_ids_query(self._team_id)})\n                WHERE distinct_id = %({id_conditions_param})s\n            )\n            '
            try:
                UUID(self._filter.search)
            except ValueError:
                pass
            else:
                id_conditions_sql = f'(id = %({id_conditions_param})s OR {id_conditions_sql})'
            prop_group = PropertyGroup(type=PropertyOperatorType.AND, values=[Property(key='email', operator='icontains', value=self._filter.search, type='person')])
            (finalization_conditions_sql, params) = parse_prop_grouped_clauses(team_id=self._team_id, property_group=prop_group, prepend=f'search_fin_{prepend}', has_person_id_joined=False, group_properties_joined=False, person_properties_mode=PersonPropertiesMode.DIRECT, _top_level=False, hogql_context=self._filter.hogql_context)
            finalization_sql = f'AND ({finalization_conditions_sql} OR {id_conditions_sql})'
            (prefiltering_conditions_sql, prefiltering_params) = parse_prop_grouped_clauses(team_id=self._team_id, property_group=prop_group, prepend=f'search_pre_{prepend}', has_person_id_joined=False, group_properties_joined=False, person_properties_mode=PersonPropertiesMode.DIRECT_ON_PERSONS, _top_level=False, hogql_context=self._filter.hogql_context)
            params.update(prefiltering_params)
            prefiltering_sql = f'AND ({prefiltering_conditions_sql} OR {id_conditions_sql})'
            params.update({id_conditions_param: self._filter.search})
            return (prefiltering_sql, finalization_sql, params)
        return ('', '', {})

    def _get_distinct_id_clause(self) -> Tuple[str, Dict]:
        if False:
            i = 10
            return i + 15
        if not isinstance(self._filter, Filter):
            return ('', {})
        if self._filter.distinct_id:
            distinct_id_clause = f'\n            AND id IN (\n                SELECT person_id FROM ({get_team_distinct_ids_query(self._team_id)}) where distinct_id = %(distinct_id_filter)s\n            )\n            '
            return (distinct_id_clause, {'distinct_id_filter': self._filter.distinct_id})
        return ('', {})

    def _add_distinct_id_join_if_needed(self, query: str, params: Dict[Any, Any]) -> Tuple[str, Dict[Any, Any]]:
        if False:
            return 10
        if not self._include_distinct_ids:
            return (query, params)
        return ('\n        SELECT person.*, groupArray(pdi.distinct_id) as distinct_ids\n        FROM ({person_query}) person\n        LEFT JOIN ({distinct_id_query}) as pdi ON person.id=pdi.person_id\n        GROUP BY person.*\n        ORDER BY created_at desc, id desc\n        '.format(person_query=query, distinct_id_query=get_team_distinct_ids_query(self._team_id)), params)

    def _get_email_clause(self) -> Tuple[str, Dict]:
        if False:
            while True:
                i = 10
        if not isinstance(self._filter, Filter):
            return ('', {})
        if self._filter.email:
            return prop_filter_json_extract(Property(key='email', value=self._filter.email, type='person'), 0, prepend='_email')
        return ('', {})

    def _get_updated_after_clause(self) -> Tuple[str, Dict]:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self._filter, Filter):
            return ('', {})
        if self._filter.updated_after:
            return ('and max(_timestamp) > parseDateTimeBestEffort(%(updated_after)s)', {'updated_after': self._filter.updated_after})
        return ('', {})