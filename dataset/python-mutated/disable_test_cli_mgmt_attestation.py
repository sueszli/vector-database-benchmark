import unittest
import azure.mgmt.attestation
from devtools_testutils import AzureMgmtTestCase, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtAttestationTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MgmtAttestationTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.attestation.AttestationManagementClient)

    @unittest.skip('skip test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_attestation(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        PROVIDER_NAME = 'myprovider6'
        CREATION_PARAMS = {'properties': {'attestation_policy': 'SgxDisableDebugMode'}, 'location': 'eastus'}
        result = self.mgmt_client.attestation_providers.create(resource_group_name=RESOURCE_GROUP, provider_name=PROVIDER_NAME, creation_params=CREATION_PARAMS)
        result = self.mgmt_client.attestation_providers.get(resource_group_name=RESOURCE_GROUP, provider_name=PROVIDER_NAME)
        result = self.mgmt_client.operations.list()
        TAGS = {'property1': 'Value1', 'property2': 'Value2', 'property3': 'Value3'}
        result = self.mgmt_client.attestation_providers.update(resource_group_name=RESOURCE_GROUP, provider_name=PROVIDER_NAME, tags=TAGS)
        result = self.mgmt_client.attestation_providers.delete(resource_group_name=RESOURCE_GROUP, provider_name=PROVIDER_NAME)
if __name__ == '__main__':
    unittest.main()