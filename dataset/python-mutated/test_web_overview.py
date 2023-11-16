from posthog.hogql_queries.web_analytics.web_overview import WebOverviewQueryRunner
from posthog.schema import WebOverviewQuery
from posthog.test.base import APIBaseTest, ClickhouseTestMixin

class TestWebOverviewQueryRunner(ClickhouseTestMixin, APIBaseTest):

    def _create_runner(self, query: WebOverviewQuery) -> WebOverviewQueryRunner:
        if False:
            print('Hello World!')
        return WebOverviewQueryRunner(team=self.team, query=query)

    def test_no_crash_when_no_data(self):
        if False:
            print('Hello World!')
        response = self._create_runner(WebOverviewQuery(kind='WebOverviewQuery', properties=[])).calculate()
        self.assertEqual(5, len(response.results))