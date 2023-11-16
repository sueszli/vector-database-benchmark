import azure.mgmt.datalake.analytics.account
import unittest
from devtools_testutils import AzureMgmtRecordedTestCase, ResourceGroupPreparer, recorded_by_proxy

class TestMgmtDatalake(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.client = self.create_mgmt_client(azure.mgmt.datalake.analytics.account.DataLakeAnalyticsAccountManagementClient)

    @recorded_by_proxy
    def test_generate_recommendations(self):
        if False:
            return 10
        response = self.client.operations.list()
        assert response
if __name__ == '__main__':
    unittest.main()