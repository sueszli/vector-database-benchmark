from typing import List, Optional, cast
from posthog.hogql import ast
from posthog.hogql.parser import parse_expr, parse_select
from posthog.hogql_queries.utils.query_date_range import QueryDateRange
from posthog.schema import ActionsNode, EventsNode

class QueryAlternator:
    """Allows query_builder to modify the query without having to expost the whole AST interface"""
    _query: ast.SelectQuery
    _selects: List[ast.Expr]
    _group_bys: List[ast.Expr]
    _select_from: ast.JoinExpr | None

    def __init__(self, query: ast.SelectQuery | ast.SelectUnionQuery):
        if False:
            print('Hello World!')
        assert isinstance(query, ast.SelectQuery)
        self._query = query
        self._selects = []
        self._group_bys = []
        self._select_from = None

    def build(self) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            while True:
                i = 10
        if len(self._selects) > 0:
            self._query.select.extend(self._selects)
        if len(self._group_bys) > 0:
            if self._query.group_by is None:
                self._query.group_by = self._group_bys
            else:
                self._query.group_by.extend(self._group_bys)
        if self._select_from is not None:
            self._query.select_from = self._select_from
        return self._query

    def append_select(self, expr: ast.Expr) -> None:
        if False:
            i = 10
            return i + 15
        self._selects.append(expr)

    def append_group_by(self, expr: ast.Expr) -> None:
        if False:
            print('Hello World!')
        self._group_bys.append(expr)

    def replace_select_from(self, join_expr: ast.JoinExpr) -> None:
        if False:
            return 10
        self._select_from = join_expr

class AggregationOperations:
    series: EventsNode | ActionsNode
    query_date_range: QueryDateRange

    def __init__(self, series: EventsNode | ActionsNode, query_date_range: QueryDateRange) -> None:
        if False:
            return 10
        self.series = series
        self.query_date_range = query_date_range

    def select_aggregation(self) -> ast.Expr:
        if False:
            for i in range(10):
                print('nop')
        if self.series.math == 'hogql' and self.series.math_hogql is not None:
            return parse_expr(self.series.math_hogql)
        elif self.series.math == 'total':
            return parse_expr('count(e.uuid)')
        elif self.series.math == 'dau':
            return parse_expr('count(DISTINCT e.person_id)')
        elif self.series.math == 'weekly_active':
            return ast.Placeholder(field='replaced')
        elif self.series.math == 'monthly_active':
            return ast.Placeholder(field='replaced')
        elif self.series.math == 'unique_session':
            return parse_expr('count(DISTINCT e."$session_id")')
        elif self.series.math == 'unique_group' and self.series.math_group_type_index is not None:
            return parse_expr(f'count(DISTINCT e."$group_{self.series.math_group_type_index}")')
        elif self.series.math_property is not None:
            if self.series.math == 'avg':
                return self._math_func('avg', None)
            elif self.series.math == 'sum':
                return self._math_func('sum', None)
            elif self.series.math == 'min':
                return self._math_func('min', None)
            elif self.series.math == 'max':
                return self._math_func('max', None)
            elif self.series.math == 'median':
                return self._math_func('median', None)
            elif self.series.math == 'p90':
                return self._math_quantile(0.9, None)
            elif self.series.math == 'p95':
                return self._math_quantile(0.95, None)
            elif self.series.math == 'p99':
                return self._math_quantile(0.99, None)
            else:
                raise NotImplementedError()
        return parse_expr('count(e.uuid)')

    def requires_query_orchestration(self) -> bool:
        if False:
            while True:
                i = 10
        math_to_return_true = ['weekly_active', 'monthly_active']
        return self._is_count_per_actor_variant() or self.series.math in math_to_return_true

    def _is_count_per_actor_variant(self):
        if False:
            for i in range(10):
                print('nop')
        return self.series.math in ['avg_count_per_actor', 'min_count_per_actor', 'max_count_per_actor', 'median_count_per_actor', 'p90_count_per_actor', 'p95_count_per_actor', 'p99_count_per_actor']

    def _math_func(self, method: str, override_chain: Optional[List[str | int]]) -> ast.Call:
        if False:
            for i in range(10):
                print('nop')
        if override_chain is not None:
            return ast.Call(name=method, args=[ast.Field(chain=override_chain)])
        if self.series.math_property == '$time':
            return ast.Call(name=method, args=[ast.Call(name='toUnixTimestamp', args=[ast.Field(chain=['properties', '$time'])])])
        if self.series.math_property == '$session_duration':
            chain = ['session', 'duration']
        else:
            chain = ['properties', self.series.math_property]
        return ast.Call(name=method, args=[ast.Field(chain=chain)])

    def _math_quantile(self, percentile: float, override_chain: Optional[List[str | int]]) -> ast.Call:
        if False:
            print('Hello World!')
        chain = ['properties', self.series.math_property]
        return ast.Call(name='quantile', params=[ast.Constant(value=percentile)], args=[ast.Field(chain=override_chain or chain)])

    def _interval_placeholders(self):
        if False:
            i = 10
            return i + 15
        if self.series.math == 'weekly_active':
            return {'exclusive_lookback': ast.Call(name='toIntervalDay', args=[ast.Constant(value=6)]), 'inclusive_lookback': ast.Call(name='toIntervalDay', args=[ast.Constant(value=7)])}
        elif self.series.math == 'monthly_active':
            return {'exclusive_lookback': ast.Call(name='toIntervalDay', args=[ast.Constant(value=29)]), 'inclusive_lookback': ast.Call(name='toIntervalDay', args=[ast.Constant(value=30)])}
        raise NotImplementedError()

    def _parent_select_query(self, inner_query: ast.SelectQuery | ast.SelectUnionQuery) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            i = 10
            return i + 15
        if self._is_count_per_actor_variant():
            return parse_select('SELECT total, day_start FROM {inner_query}', placeholders={'inner_query': inner_query})
        return parse_select('\n                SELECT\n                    counts AS total,\n                    dateTrunc({interval}, timestamp) AS day_start\n                FROM {inner_query}\n                WHERE timestamp >= {date_from} AND timestamp <= {date_to}\n            ', placeholders={**self.query_date_range.to_placeholders(), 'inner_query': inner_query})

    def _inner_select_query(self, cross_join_select_query: ast.SelectQuery | ast.SelectUnionQuery) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            for i in range(10):
                print('nop')
        if self._is_count_per_actor_variant():
            if self.series.math == 'avg_count_per_actor':
                math_func = self._math_func('avg', ['total'])
            elif self.series.math == 'min_count_per_actor':
                math_func = self._math_func('min', ['total'])
            elif self.series.math == 'max_count_per_actor':
                math_func = self._math_func('max', ['total'])
            elif self.series.math == 'median_count_per_actor':
                math_func = self._math_func('median', ['total'])
            elif self.series.math == 'p90_count_per_actor':
                math_func = self._math_quantile(0.9, ['total'])
            elif self.series.math == 'p95_count_per_actor':
                math_func = self._math_quantile(0.95, ['total'])
            elif self.series.math == 'p99_count_per_actor':
                math_func = self._math_quantile(0.99, ['total'])
            else:
                raise NotImplementedError()
            total_alias = ast.Alias(alias='total', expr=math_func)
            return parse_select('\n                    SELECT\n                        {total_alias}, day_start\n                    FROM {inner_query}\n                    GROUP BY day_start\n                ', placeholders={'inner_query': cross_join_select_query, 'total_alias': total_alias})
        return parse_select("\n                SELECT\n                    d.timestamp,\n                    COUNT(DISTINCT actor_id) AS counts\n                FROM (\n                    SELECT\n                        toStartOfDay({date_to}) - toIntervalDay(number) AS timestamp\n                    FROM\n                        numbers(dateDiff('day', toStartOfDay({date_from} - {inclusive_lookback}), {date_to}))\n                ) d\n                CROSS JOIN {cross_join_select_query} e\n                WHERE\n                    e.timestamp <= d.timestamp + INTERVAL 1 DAY AND\n                    e.timestamp > d.timestamp - {exclusive_lookback}\n                GROUP BY d.timestamp\n                ORDER BY d.timestamp\n            ", placeholders={**self.query_date_range.to_placeholders(), **self._interval_placeholders(), 'cross_join_select_query': cross_join_select_query})

    def _events_query(self, events_where_clause: ast.Expr, sample_value: ast.RatioExpr) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            return 10
        if self._is_count_per_actor_variant():
            return parse_select('\n                    SELECT\n                        count(e.uuid) AS total,\n                        dateTrunc({interval}, timestamp) AS day_start\n                    FROM events AS e\n                    SAMPLE {sample}\n                    WHERE {events_where_clause}\n                    GROUP BY e.person_id, day_start\n                ', placeholders={**self.query_date_range.to_placeholders(), 'events_where_clause': events_where_clause, 'sample': sample_value})
        return parse_select('\n                SELECT\n                    timestamp as timestamp,\n                    e.person_id AS actor_id\n                FROM\n                    events e\n                SAMPLE {sample}\n                WHERE {events_where_clause}\n                GROUP BY\n                    timestamp,\n                    actor_id\n            ', placeholders={'events_where_clause': events_where_clause, 'sample': sample_value})

    def get_query_orchestrator(self, events_where_clause: ast.Expr, sample_value: ast.RatioExpr):
        if False:
            while True:
                i = 10
        events_query = cast(ast.SelectQuery, self._events_query(events_where_clause, sample_value))
        inner_select = cast(ast.SelectQuery, self._inner_select_query(events_query))
        parent_select = cast(ast.SelectQuery, self._parent_select_query(inner_select))

        class QueryOrchestrator:
            events_query_builder: QueryAlternator
            inner_select_query_builder: QueryAlternator
            parent_select_query_builder: QueryAlternator

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.events_query_builder = QueryAlternator(events_query)
                self.inner_select_query_builder = QueryAlternator(inner_select)
                self.parent_select_query_builder = QueryAlternator(parent_select)

            def build(self):
                if False:
                    return 10
                self.events_query_builder.build()
                self.inner_select_query_builder.build()
                self.parent_select_query_builder.build()
                return parent_select
        return QueryOrchestrator()