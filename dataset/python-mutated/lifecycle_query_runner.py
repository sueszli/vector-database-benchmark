from datetime import timedelta
from math import ceil
from typing import Optional, Any, Dict, List
from django.utils.timezone import datetime
from posthog.caching.insights_api import BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL, REDUCED_MINIMUM_INSIGHT_REFRESH_INTERVAL
from posthog.caching.utils import is_stale
from posthog.hogql import ast
from posthog.hogql.parser import parse_expr, parse_select
from posthog.hogql.printer import to_printed_hogql
from posthog.hogql.property import property_to_expr, action_to_expr
from posthog.hogql.query import execute_hogql_query
from posthog.hogql.timings import HogQLTimings
from posthog.hogql_queries.query_runner import QueryRunner
from posthog.models import Team, Action
from posthog.hogql_queries.utils.query_date_range import QueryDateRange
from posthog.models.filters.mixins.utils import cached_property
from posthog.schema import LifecycleQuery, ActionsNode, EventsNode, LifecycleQueryResponse

class LifecycleQueryRunner(QueryRunner):
    query: LifecycleQuery
    query_type = LifecycleQuery

    def __init__(self, query: LifecycleQuery | Dict[str, Any], team: Team, timings: Optional[HogQLTimings]=None, in_export_context: Optional[bool]=False):
        if False:
            print('Hello World!')
        super().__init__(query, team, timings, in_export_context)

    def to_query(self) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            while True:
                i = 10
        if self.query.samplingFactor == 0:
            counts_with_sampling = ast.Constant(value=0)
        elif self.query.samplingFactor is not None and self.query.samplingFactor != 1:
            counts_with_sampling = parse_expr('round(counts * (1 / {sampling_factor}))', {'sampling_factor': ast.Constant(value=self.query.samplingFactor)})
        else:
            counts_with_sampling = parse_expr('counts')
        placeholders = {**self.query_date_range.to_placeholders(), 'events_query': self.events_query, 'periods_query': self.periods_query, 'counts_with_sampling': counts_with_sampling}
        with self.timings.measure('lifecycle_query'):
            lifecycle_query = parse_select("\n                    SELECT groupArray(start_of_period) AS date,\n                           groupArray({counts_with_sampling}) AS total,\n                           status\n                    FROM (\n                        SELECT\n                            status = 'dormant' ? negate(sum(counts)) : negate(negate(sum(counts))) as counts,\n                            start_of_period,\n                            status\n                        FROM (\n                            SELECT\n                                periods.start_of_period as start_of_period,\n                                0 AS counts,\n                                status\n                            FROM {periods_query} as periods\n                            CROSS JOIN (\n                                SELECT status\n                                FROM (SELECT 1)\n                                ARRAY JOIN ['new', 'returning', 'resurrecting', 'dormant'] as status\n                            ) as sec\n                            ORDER BY status, start_of_period\n                            UNION ALL\n                            SELECT\n                                start_of_period, count(DISTINCT person_id) AS counts, status\n                            FROM {events_query}\n                            GROUP BY start_of_period, status\n                        )\n                        WHERE start_of_period <= dateTrunc({interval}, {date_to})\n                            AND start_of_period >= dateTrunc({interval}, {date_from})\n                        GROUP BY start_of_period, status\n                        ORDER BY start_of_period ASC\n                    )\n                    GROUP BY status\n                ", placeholders, timings=self.timings)
        return lifecycle_query

    def to_persons_query(self, day: Optional[str]=None, status: Optional[str]=None) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            for i in range(10):
                print('nop')
        with self.timings.measure('persons_query'):
            exprs = []
            if day is not None:
                exprs.append(ast.CompareOperation(op=ast.CompareOperationOp.Eq, left=ast.Field(chain=['start_of_period']), right=ast.Constant(value=day)))
            if status is not None:
                exprs.append(ast.CompareOperation(op=ast.CompareOperationOp.Eq, left=ast.Field(chain=['status']), right=ast.Constant(value=status)))
            return parse_select('SELECT person_id FROM {events_query} WHERE {where}', placeholders={'events_query': self.events_query, 'where': ast.And(exprs=exprs) if len(exprs) > 0 else ast.Constant(value=1)})

    def calculate(self) -> LifecycleQueryResponse:
        if False:
            i = 10
            return i + 15
        query = self.to_query()
        hogql = to_printed_hogql(query, self.team.pk)
        response = execute_hogql_query(query_type='LifecycleQuery', query=query, team=self.team, timings=self.timings)
        order = {'new': 1, 'returning': 2, 'resurrecting': 3, 'dormant': 4}
        results = sorted(response.results, key=lambda result: order.get(result[2], 5))
        res = []
        for val in results:
            counts = val[1]
            labels = [item.strftime('%-d-%b-%Y{}'.format(' %H:%M' if self.query_date_range.interval_name == 'hour' else '')) for item in val[0]]
            days = [item.strftime('%Y-%m-%d{}'.format(' %H:%M:%S' if self.query_date_range.interval_name == 'hour' else '')) for item in val[0]]
            action_object = {}
            label = '{} - {}'.format('', val[2])
            if isinstance(self.query.series[0], ActionsNode):
                action = Action.objects.get(pk=int(self.query.series[0].id), team=self.team)
                label = '{} - {}'.format(action.name, val[2])
                action_object = {'id': str(action.pk), 'name': action.name, 'type': 'actions', 'order': 0, 'math': 'total'}
            elif isinstance(self.query.series[0], EventsNode):
                event = self.query.series[0].event
                label = '{} - {}'.format('All events' if event is None else event, val[2])
                action_object = {'id': event, 'name': 'All events' if event is None else event, 'type': 'events', 'order': 0, 'math': 'total'}
            additional_values = {'label': label, 'status': val[2]}
            res.append({'action': action_object, 'data': [float(c) for c in counts], 'count': float(sum(counts)), 'labels': labels, 'days': days, **additional_values})
        return LifecycleQueryResponse(results=res, timings=response.timings, hogql=hogql)

    @cached_property
    def query_date_range(self):
        if False:
            return 10
        return QueryDateRange(date_range=self.query.dateRange, team=self.team, interval=self.query.interval, now=datetime.now())

    @cached_property
    def event_filter(self) -> ast.Expr:
        if False:
            print('Hello World!')
        event_filters: List[ast.Expr] = []
        with self.timings.measure('date_range'):
            event_filters.append(parse_expr('timestamp >= dateTrunc({interval}, {date_from}) - {one_interval}', {'interval': self.query_date_range.interval_period_string_as_hogql_constant(), 'one_interval': self.query_date_range.one_interval_period(), 'date_from': self.query_date_range.date_from_as_hogql()}, timings=self.timings))
            event_filters.append(parse_expr('timestamp < dateTrunc({interval}, {date_to}) + {one_interval}', {'interval': self.query_date_range.interval_period_string_as_hogql_constant(), 'one_interval': self.query_date_range.one_interval_period(), 'date_to': self.query_date_range.date_to_as_hogql()}, timings=self.timings))
        with self.timings.measure('properties'):
            if self.query.properties is not None and self.query.properties != []:
                event_filters.append(property_to_expr(self.query.properties, self.team))
        with self.timings.measure('series_filters'):
            for serie in self.query.series or []:
                if isinstance(serie, ActionsNode):
                    action = Action.objects.get(pk=int(serie.id), team=self.team)
                    event_filters.append(action_to_expr(action))
                elif isinstance(serie, EventsNode):
                    if serie.event is not None:
                        event_filters.append(ast.CompareOperation(op=ast.CompareOperationOp.Eq, left=ast.Field(chain=['event']), right=ast.Constant(value=str(serie.event))))
                else:
                    raise ValueError(f'Invalid serie kind: {serie.kind}')
                if serie.properties is not None and serie.properties != []:
                    event_filters.append(property_to_expr(serie.properties, self.team))
        with self.timings.measure('test_account_filters'):
            if self.query.filterTestAccounts and isinstance(self.team.test_account_filters, list) and (len(self.team.test_account_filters) > 0):
                for property in self.team.test_account_filters:
                    event_filters.append(property_to_expr(property, self.team))
        if len(event_filters) == 0:
            return ast.Constant(value=True)
        elif len(event_filters) == 1:
            return event_filters[0]
        else:
            return ast.And(exprs=event_filters)

    @cached_property
    def events_query(self):
        if False:
            while True:
                i = 10
        with self.timings.measure('events_query'):
            events_query = parse_select("\n                    SELECT\n                        events.person.id as person_id,\n                        min(events.person.created_at) AS created_at,\n                        arraySort(groupUniqArray(dateTrunc({interval}, events.timestamp))) AS all_activity,\n                        arrayPopBack(arrayPushFront(all_activity, dateTrunc({interval}, created_at))) as previous_activity,\n                        arrayPopFront(arrayPushBack(all_activity, dateTrunc({interval}, toDateTime('1970-01-01 00:00:00')))) as following_activity,\n                        arrayMap((previous, current, index) -> (previous = current ? 'new' : ((current - {one_interval_period}) = previous AND index != 1) ? 'returning' : 'resurrecting'), previous_activity, all_activity, arrayEnumerate(all_activity)) as initial_status,\n                        arrayMap((current, next) -> (current + {one_interval_period} = next ? '' : 'dormant'), all_activity, following_activity) as dormant_status,\n                        arrayMap(x -> x + {one_interval_period}, arrayFilter((current, is_dormant) -> is_dormant = 'dormant', all_activity, dormant_status)) as dormant_periods,\n                        arrayMap(x -> 'dormant', dormant_periods) as dormant_label,\n                        arrayConcat(arrayZip(all_activity, initial_status), arrayZip(dormant_periods, dormant_label)) as temp_concat,\n                        arrayJoin(temp_concat) as period_status_pairs,\n                        period_status_pairs.1 as start_of_period,\n                        period_status_pairs.2 as status\n                    FROM events\n                    WHERE {event_filter}\n                    GROUP BY person_id\n                ", placeholders={**self.query_date_range.to_placeholders(), 'event_filter': self.event_filter}, timings=self.timings)
            sampling_factor = self.query.samplingFactor
            if sampling_factor is not None and isinstance(sampling_factor, float):
                sample_expr = ast.SampleExpr(sample_value=ast.RatioExpr(left=ast.Constant(value=sampling_factor)))
                events_query.select_from.sample = sample_expr
        return events_query

    @cached_property
    def periods_query(self):
        if False:
            while True:
                i = 10
        with self.timings.measure('periods_query'):
            periods_query = parse_select('\n                    SELECT (\n                        dateTrunc({interval}, {date_to}) - {number_interval_period}\n                    ) AS start_of_period\n                    FROM numbers(\n                        dateDiff(\n                            {interval},\n                            dateTrunc({interval}, {date_from}),\n                            dateTrunc({interval}, {date_to} + {one_interval_period})\n                        )\n                    )\n                ', placeholders=self.query_date_range.to_placeholders(), timings=self.timings)
        return periods_query

    def _is_stale(self, cached_result_package):
        if False:
            print('Hello World!')
        date_to = self.query_date_range.date_to()
        interval = self.query_date_range.interval_name
        return is_stale(self.team, date_to, interval, cached_result_package)

    def _refresh_frequency(self):
        if False:
            i = 10
            return i + 15
        date_to = self.query_date_range.date_to()
        date_from = self.query_date_range.date_from()
        interval = self.query_date_range.interval_name
        delta_days: Optional[int] = None
        if date_from and date_to:
            delta = date_to - date_from
            delta_days = ceil(delta.total_seconds() / timedelta(days=1).total_seconds())
        refresh_frequency = BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL
        if interval == 'hour' or (delta_days is not None and delta_days <= 7):
            refresh_frequency = REDUCED_MINIMUM_INSIGHT_REFRESH_INTERVAL
        return refresh_frequency