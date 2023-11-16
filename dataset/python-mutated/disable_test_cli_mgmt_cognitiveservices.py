import unittest
import azure.mgmt.cognitiveservices
from devtools_testutils import AzureMgmtTestCase, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtCognitiveServicesTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(MgmtCognitiveServicesTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.cognitiveservices.CognitiveServicesManagementClient)

    @unittest.skip('hard to test')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_cognitiveservices(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        ACCOUNT_NAME = 'myAccount'
        LOCATION = 'myLocation'
        BODY = {'location': 'West US', 'kind': 'CognitiveServices', 'sku': {'name': 'S0'}, 'identity': {'type': 'SystemAssigned'}}
        result = self.mgmt_client.accounts.create(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, account=BODY)
        BODY = {'location': 'West US', 'kind': 'Emotion', 'sku': {'name': 'S0'}, 'properties': {'encryption': {'key_vault_properties': {'key_name': 'KeyName', 'key_version': '891CF236-D241-4738-9462-D506AF493DFA', 'key_vault_uri': 'https://pltfrmscrts-use-pc-dev.vault.azure.net/'}, 'key_source': 'Microsoft.KeyVault'}, 'user_owned_storage': [{'resource_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Storage/storageAccountsfelixwatest'}]}, 'identity': {'type': 'SystemAssigned'}}
        result = self.mgmt_client.accounts.get_usages(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.accounts.list_skus(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.accounts.get_properties(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        result = self.mgmt_client.accounts.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.accounts.list()
        result = self.mgmt_client.resource_skus.list()
        result = self.mgmt_client.operations.list()
        result = self.mgmt_client.accounts.regenerate_key(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME, key_name='Key2')
        result = self.mgmt_client.accounts.list_keys(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
        BODY = {'sku': {'name': 'S2'}}
        SKUS = ['S0']
        result = self.mgmt_client.check_sku_availability(location='eastus', skus=SKUS, kind='Face', type='Microsoft.CognitiveServices/accounts')
        result = self.mgmt_client.accounts.delete(resource_group_name=RESOURCE_GROUP, account_name=ACCOUNT_NAME)
if __name__ == '__main__':
    unittest.main()