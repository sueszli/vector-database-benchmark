from django.utils.timezone import datetime
from posthog.hogql import ast
from posthog.hogql.parser import parse_select
from posthog.hogql.query import execute_hogql_query
from posthog.hogql_queries.utils.query_date_range import QueryDateRange
from posthog.hogql_queries.web_analytics.web_analytics_query_runner import WebAnalyticsQueryRunner
from posthog.models.filters.mixins.utils import cached_property
from posthog.schema import WebTopClicksQuery, WebTopClicksQueryResponse

class WebTopClicksQueryRunner(WebAnalyticsQueryRunner):
    query: WebTopClicksQuery
    query_type = WebTopClicksQuery

    def to_query(self) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            return 10
        with self.timings.measure('top_clicks_query'):
            top_sources_query = parse_select("\nSELECT\n    properties.$el_text as el_text,\n    count() as total_clicks,\n    COUNT(DISTINCT events.person_id) as unique_visitors\nFROM\n    events\nWHERE\n    event == '$autocapture'\nAND events.properties.$event_type = 'click'\nAND el_text IS NOT NULL\nAND ({events_where})\nGROUP BY\n    el_text\nORDER BY total_clicks DESC\nLIMIT 10\n                ", timings=self.timings, placeholders={'event_properties': self.events_where(), 'date_from': self.query_date_range.date_from_as_hogql(), 'date_to': self.query_date_range.date_to_as_hogql()})
        return top_sources_query

    def calculate(self):
        if False:
            while True:
                i = 10
        response = execute_hogql_query(query_type='top_sources_query', query=self.to_query(), team=self.team, timings=self.timings)
        return WebTopClicksQueryResponse(columns=response.columns, results=response.results, timings=response.timings, types=response.types)

    @cached_property
    def query_date_range(self):
        if False:
            for i in range(10):
                print('nop')
        return QueryDateRange(date_range=self.query.dateRange, team=self.team, interval=None, now=datetime.now())