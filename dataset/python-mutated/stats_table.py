from posthog.hogql import ast
from posthog.hogql.parser import parse_select, parse_expr
from posthog.hogql.query import execute_hogql_query
from posthog.hogql_queries.web_analytics.ctes import COUNTS_CTE, BOUNCE_RATE_CTE
from posthog.hogql_queries.web_analytics.web_analytics_query_runner import WebAnalyticsQueryRunner
from posthog.schema import WebStatsTableQuery, WebStatsBreakdown, WebStatsTableQueryResponse

class WebStatsTableQueryRunner(WebAnalyticsQueryRunner):
    query: WebStatsTableQuery
    query_type = WebStatsTableQuery

    def to_query(self) -> ast.SelectQuery | ast.SelectUnionQuery:
        if False:
            return 10
        with self.timings.measure('bounce_rate_query'):
            bounce_rate_query = parse_select(BOUNCE_RATE_CTE, timings=self.timings, placeholders={'session_where': self.session_where(), 'session_having': self.session_having(), 'breakdown_by': self.bounce_breakdown()}, backend='cpp')
        with self.timings.measure('counts_query'):
            counts_query = parse_select(COUNTS_CTE, timings=self.timings, placeholders={'counts_where': self.events_where(), 'breakdown_by': self.counts_breakdown()}, backend='cpp')
        with self.timings.measure('top_pages_query'):
            top_sources_query = parse_select('\nSELECT\n    counts.breakdown_value as "context.columns.breakdown_value",\n    counts.total_pageviews as "context.columns.views",\n    counts.unique_visitors as "context.columns.visitors",\n    bounce_rate.bounce_rate as "context.columns.bounce_rate"\nFROM\n    {counts_query} AS counts\nLEFT OUTER JOIN\n    {bounce_rate_query} AS bounce_rate\nON\n    counts.breakdown_value = bounce_rate.breakdown_value\nWHERE\n    {where_breakdown}\nORDER BY\n    "context.columns.views" DESC\nLIMIT 10\n                ', timings=self.timings, placeholders={'counts_query': counts_query, 'bounce_rate_query': bounce_rate_query, 'where_breakdown': self.where_breakdown()}, backend='cpp')
        return top_sources_query

    def calculate(self):
        if False:
            print('Hello World!')
        response = execute_hogql_query(query_type='top_sources_query', query=self.to_query(), team=self.team, timings=self.timings)
        return WebStatsTableQueryResponse(columns=response.columns, results=response.results, timings=response.timings, types=response.types, hogql=response.hogql)

    def counts_breakdown(self):
        if False:
            for i in range(10):
                print('nop')
        match self.query.breakdownBy:
            case WebStatsBreakdown.Page:
                return ast.Field(chain=['properties', '$pathname'])
            case WebStatsBreakdown.InitialPage:
                return ast.Field(chain=['person', 'properties', '$initial_pathname'])
            case WebStatsBreakdown.InitialReferringDomain:
                return ast.Field(chain=['person', 'properties', '$initial_referring_domain'])
            case WebStatsBreakdown.InitialUTMSource:
                return ast.Field(chain=['person', 'properties', '$initial_utm_source'])
            case WebStatsBreakdown.InitialUTMCampaign:
                return ast.Field(chain=['person', 'properties', '$initial_utm_campaign'])
            case WebStatsBreakdown.InitialUTMMedium:
                return ast.Field(chain=['person', 'properties', '$initial_utm_medium'])
            case WebStatsBreakdown.InitialUTMTerm:
                return ast.Field(chain=['person', 'properties', '$initial_utm_term'])
            case WebStatsBreakdown.InitialUTMContent:
                return ast.Field(chain=['person', 'properties', '$initial_utm_content'])
            case WebStatsBreakdown.Browser:
                return ast.Field(chain=['properties', '$browser'])
            case WebStatsBreakdown.OS:
                return ast.Field(chain=['properties', '$os'])
            case WebStatsBreakdown.DeviceType:
                return ast.Field(chain=['properties', '$device_type'])
            case WebStatsBreakdown.Country:
                return ast.Field(chain=['properties', '$geoip_country_code'])
            case WebStatsBreakdown.Region:
                return parse_expr('tuple(properties.$geoip_country_code, properties.$geoip_subdivision_1_code, properties.$geoip_subdivision_1_name)')
            case WebStatsBreakdown.City:
                return parse_expr('tuple(properties.$geoip_country_code, properties.$geoip_city_name)')
            case _:
                raise NotImplementedError('Breakdown not implemented')

    def bounce_breakdown(self):
        if False:
            return 10
        match self.query.breakdownBy:
            case WebStatsBreakdown.Page:
                return ast.Call(name='any', args=[ast.Field(chain=['person', 'properties', '$initial_pathname'])])
            case _:
                return ast.Call(name='any', args=[self.counts_breakdown()])

    def where_breakdown(self):
        if False:
            for i in range(10):
                print('nop')
        match self.query.breakdownBy:
            case WebStatsBreakdown.Region:
                return parse_expr('tupleElement("context.columns.breakdown_value", 2) IS NOT NULL')
            case WebStatsBreakdown.City:
                return parse_expr('tupleElement("context.columns.breakdown_value", 2) IS NOT NULL')
            case WebStatsBreakdown.InitialUTMSource:
                return parse_expr('TRUE')
            case WebStatsBreakdown.InitialUTMCampaign:
                return parse_expr('TRUE')
            case WebStatsBreakdown.InitialUTMMedium:
                return parse_expr('TRUE')
            case WebStatsBreakdown.InitialUTMTerm:
                return parse_expr('TRUE')
            case WebStatsBreakdown.InitialUTMContent:
                return parse_expr('TRUE')
            case _:
                return parse_expr('"context.columns.breakdown_value" IS NOT NULL')