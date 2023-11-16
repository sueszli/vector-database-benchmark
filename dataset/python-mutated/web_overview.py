from typing import Optional
from django.utils.timezone import datetime
from posthog.hogql import ast
from posthog.hogql.parser import parse_select
from posthog.hogql.property import property_to_expr
from posthog.hogql.query import execute_hogql_query
from posthog.hogql_queries.utils.query_date_range import QueryDateRange
from posthog.hogql_queries.web_analytics.web_analytics_query_runner import WebAnalyticsQueryRunner
from posthog.models.filters.mixins.utils import cached_property
from posthog.schema import WebOverviewQueryResponse, WebOverviewQuery

class WebOverviewQueryRunner(WebAnalyticsQueryRunner):
    query: WebOverviewQuery
    query_type = WebOverviewQuery

    def to_query(self) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            print('Hello World!')
        with self.timings.measure('date_expr'):
            start = self.query_date_range.previous_period_date_from_as_hogql()
            mid = self.query_date_range.date_from_as_hogql()
            end = self.query_date_range.date_to_as_hogql()
        with self.timings.measure('overview_stats_query'):
            query = parse_select("\nWITH pages_query AS (\n        SELECT\n        uniq(if(timestamp >= {mid} AND timestamp < {end}, events.person_id, NULL)) AS unique_users,\n        uniq(if(timestamp >= {start} AND timestamp < {mid}, events.person_id, NULL)) AS previous_unique_users,\n        countIf(timestamp >= {mid} AND timestamp < {end}) AS current_pageviews,\n        countIf(timestamp >= {start} AND timestamp < {mid}) AS previous_pageviews,\n        uniq(if(timestamp >= {mid} AND timestamp < {end}, events.properties.$session_id, NULL)) AS unique_sessions,\n        uniq(if(timestamp >= {start} AND timestamp < {mid}, events.properties.$session_id, NULL)) AS previous_unique_sessions\n    FROM\n        events\n    WHERE\n        event = '$pageview' AND\n        timestamp >= {start} AND\n        timestamp < {end} AND\n        {event_properties}\n    ),\nsessions_query AS (\n    SELECT\n        avg(if(min_timestamp > {mid}, duration_s, NULL)) AS avg_duration_s,\n        avg(if(min_timestamp <= {mid}, duration_s, NULL)) AS prev_avg_duration_s,\n        avg(if(min_timestamp > {mid}, is_bounce, NULL)) AS bounce_rate,\n        avg(if(min_timestamp <= {mid}, is_bounce, NULL)) AS prev_bounce_rate\n    FROM (SELECT\n            events.properties.`$session_id` AS session_id,\n            min(events.timestamp) AS min_timestamp,\n            max(events.timestamp) AS max_timestamp,\n            dateDiff('second', min_timestamp, max_timestamp) AS duration_s,\n            countIf(events.event == '$pageview') AS num_pageviews,\n            countIf(events.event == '$autocapture') AS num_autocaptures,\n\n            -- definition of a GA4 bounce from here https://support.google.com/analytics/answer/12195621?hl=en\n            (num_autocaptures == 0 AND num_pageviews <= 1 AND duration_s < 10) AS is_bounce\n        FROM\n            events\n        WHERE\n            session_id IS NOT NULL\n            AND (events.event == '$pageview' OR events.event == '$autocapture' OR events.event == '$pageleave')\n            AND ({session_where})\n        GROUP BY\n            events.properties.`$session_id`\n        HAVING\n            ({session_having})\n        )\n    )\nSELECT\n    unique_users,\n    previous_unique_users,\n    current_pageviews,\n    previous_pageviews,\n    unique_sessions,\n    previous_unique_sessions,\n    avg_duration_s,\n    prev_avg_duration_s,\n    bounce_rate,\n    prev_bounce_rate\nFROM pages_query\nCROSS JOIN sessions_query\n                ", timings=self.timings, placeholders={'start': start, 'mid': mid, 'end': end, 'event_properties': self.event_properties(), 'session_where': self.session_where(include_previous_period=True), 'session_having': self.session_having(include_previous_period=True)}, backend='cpp')
        return query

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        response = execute_hogql_query(query_type='overview_stats_pages_query', query=self.to_query(), team=self.team, timings=self.timings)
        row = response.results[0]
        return WebOverviewQueryResponse(results=[to_data('visitors', 'unit', row[0], row[1]), to_data('views', 'unit', row[2], row[3]), to_data('sessions', 'unit', row[4], row[5]), to_data('session duration', 'duration_s', row[6], row[7]), to_data('bounce rate', 'percentage', row[8], row[9], is_increase_bad=True)])

    @cached_property
    def query_date_range(self):
        if False:
            print('Hello World!')
        return QueryDateRange(date_range=self.query.dateRange, team=self.team, interval=None, now=datetime.now())

    def event_properties(self) -> ast.Expr:
        if False:
            i = 10
            return i + 15
        return property_to_expr(self.query.properties, team=self.team)

def to_data(key: str, kind: str, value: Optional[float], previous: Optional[float], is_increase_bad: Optional[bool]=None) -> dict:
    if False:
        print('Hello World!')
    if kind == 'percentage':
        if value is not None:
            value = value * 100
        if previous is not None:
            previous = previous * 100
    return {'key': key, 'kind': kind, 'isIncreaseBad': is_increase_bad, 'value': value, 'previous': previous, 'changeFromPreviousPct': round(100 * (value - previous) / previous) if value is not None and previous is not None and (previous != 0) else None}