from ee.clickhouse.queries.retention.retention_event_query import ClickhouseRetentionEventsQuery
from posthog.models.filters.mixins.utils import cached_property
from posthog.queries.retention.actors_query import RetentionActorsByPeriod

class ClickhouseRetentionActorsByPeriod(RetentionActorsByPeriod):
    _retention_events_query = ClickhouseRetentionEventsQuery

    @cached_property
    def aggregation_group_type_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self._filter.aggregation_group_type_index