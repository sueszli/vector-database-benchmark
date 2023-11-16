import pytest
from azure.mgmt.databricks import AzureDatabricksManagementClient
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

@pytest.mark.skip(reason='fix later')
class TestAzureMgmtDatabricks(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.mgmt_client = self.create_mgmt_client(AzureDatabricksManagementClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_list_by_sub(self, resource_group):
        if False:
            return 10
        self.mgmt_client.workspaces.list_by_subscription()