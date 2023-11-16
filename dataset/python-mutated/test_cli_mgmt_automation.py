import time
import unittest
import azure.mgmt.automation
from devtools_testutils import AzureMgmtRecordedTestCase, ResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtAutomationClient(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.automation.AutomationClient)

    @ResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_automation(self, resource_group):
        if False:
            while True:
                i = 10
        AUTOMATION_ACCOUNT_NAME = 'myAutomationAccount9'
        BODY = {'sku': {'name': 'Free'}, 'name': AUTOMATION_ACCOUNT_NAME, 'location': 'East US 2'}
        self.mgmt_client.automation_account.create_or_update(resource_group.name, AUTOMATION_ACCOUNT_NAME, BODY)
        self.mgmt_client.software_update_configuration_machine_runs.list(resource_group.name, AUTOMATION_ACCOUNT_NAME)
if __name__ == '__main__':
    unittest.main()