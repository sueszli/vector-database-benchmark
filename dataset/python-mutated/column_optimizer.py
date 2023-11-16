from typing import Counter as TCounter
from typing import Set, cast
from posthog.clickhouse.materialized_columns.column import ColumnName
from posthog.constants import TREND_FILTER_TYPE_ACTIONS, FunnelCorrelationType
from posthog.models.action.util import get_action_tables_and_properties
from posthog.models.filters.mixins.utils import cached_property
from posthog.models.filters.properties_timeline_filter import PropertiesTimelineFilter
from posthog.models.filters.stickiness_filter import StickinessFilter
from posthog.models.filters.utils import GroupTypeIndex
from posthog.models.property import PropertyIdentifier
from posthog.models.property.util import box_value, count_hogql_properties, extract_tables_and_properties
from posthog.queries.column_optimizer.foss_column_optimizer import FOSSColumnOptimizer
from posthog.queries.trends.util import is_series_group_based

class EnterpriseColumnOptimizer(FOSSColumnOptimizer):

    @cached_property
    def group_types_to_query(self) -> Set[GroupTypeIndex]:
        if False:
            print('Hello World!')
        used_properties = self.used_properties_with_type('group')
        return set((cast(GroupTypeIndex, group_type_index) for (_, _, group_type_index) in used_properties))

    @cached_property
    def group_on_event_columns_to_query(self) -> Set[ColumnName]:
        if False:
            return 10
        'Returns a list of event table group columns containing materialized properties that this query needs'
        used_properties = self.used_properties_with_type('group')
        columns_to_query: Set[ColumnName] = set()
        for group_type_index in range(5):
            columns_to_query = columns_to_query.union(self.columns_to_query('events', {property for property in used_properties if property[2] == group_type_index}, f'group{group_type_index}_properties'))
        return columns_to_query

    @cached_property
    def properties_used_in_filter(self) -> TCounter[PropertyIdentifier]:
        if False:
            print('Hello World!')
        'Returns collection of properties + types that this query would use'
        counter: TCounter[PropertyIdentifier] = extract_tables_and_properties(self.filter.property_groups.flat)
        if not isinstance(self.filter, StickinessFilter):
            if self.filter.breakdown_type in ['event', 'person']:
                boxed_breakdown = box_value(self.filter.breakdown)
                for b in boxed_breakdown:
                    if isinstance(b, str):
                        counter[b, self.filter.breakdown_type, self.filter.breakdown_group_type_index] += 1
            elif self.filter.breakdown_type == 'group':
                assert isinstance(self.filter.breakdown, str)
                counter[self.filter.breakdown, self.filter.breakdown_type, self.filter.breakdown_group_type_index] += 1
            elif self.filter.breakdown_type == 'hogql':
                if isinstance(self.filter.breakdown, list):
                    expr = str(self.filter.breakdown[0])
                else:
                    expr = str(self.filter.breakdown)
                counter = count_hogql_properties(expr, counter)
            for breakdown in self.filter.breakdowns or []:
                if breakdown['type'] == 'hogql':
                    counter = count_hogql_properties(breakdown['property'], counter)
                else:
                    counter[breakdown['property'], breakdown['type'], self.filter.breakdown_group_type_index] += 1
        for entity in self.entities_used_in_filter():
            counter += extract_tables_and_properties(entity.property_groups.flat)
            if entity.math_property:
                counter[entity.math_property, 'event', None] += 1
            if is_series_group_based(entity):
                counter[f'$group_{entity.math_group_type_index}', 'event', None] += 1
            if entity.math == 'unique_session':
                counter[f'$session_id', 'event', None] += 1
            if entity.type == TREND_FILTER_TYPE_ACTIONS:
                counter += get_action_tables_and_properties(entity.get_action())
        if not isinstance(self.filter, (StickinessFilter, PropertiesTimelineFilter)) and self.filter.correlation_type == FunnelCorrelationType.PROPERTIES and self.filter.correlation_property_names:
            if self.filter.aggregation_group_type_index is not None:
                for prop_value in self.filter.correlation_property_names:
                    counter[prop_value, 'group', self.filter.aggregation_group_type_index] += 1
            else:
                for prop_value in self.filter.correlation_property_names:
                    counter[prop_value, 'person', None] += 1
        return counter