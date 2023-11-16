from azure.applicationinsights import ApplicationInsightsDataClient
from azure.applicationinsights.models import QueryBody
from devtools_testutils import AzureMgmtTestCase

class ApplicationInsightsQueryTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ApplicationInsightsQueryTest, self).setUp()
        self.client = self.create_basic_client(ApplicationInsightsDataClient)

    def test_query(self):
        if False:
            return 10
        query = 'requests | take 10'
        application = 'DEMO_APP'
        result = self.client.query.execute(application, QueryBody(query=query))
        self.assertGreaterEqual(len(result.tables), 1)
        self.assertEqual(len(result.tables[0].columns), 37)
        self.assertEqual(len(result.tables[0].rows), 10)
        self.assertIs(type(result.tables[0].rows[0][7]), float)