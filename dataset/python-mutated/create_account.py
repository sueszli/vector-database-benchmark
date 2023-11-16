from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cognitiveservices\n# USAGE\n    python create_account.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CognitiveServicesManagementClient(credential=DefaultAzureCredential(), subscription_id='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')
    response = client.accounts.begin_create(resource_group_name='myResourceGroup', account_name='testCreate1', account={'identity': {'type': 'SystemAssigned'}, 'kind': 'Emotion', 'location': 'West US', 'properties': {'encryption': {'keySource': 'Microsoft.KeyVault', 'keyVaultProperties': {'keyName': 'KeyName', 'keyVaultUri': 'https://pltfrmscrts-use-pc-dev.vault.azure.net/', 'keyVersion': '891CF236-D241-4738-9462-D506AF493DFA'}}, 'userOwnedStorage': [{'resourceId': '/subscriptions/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/resourceGroups/myResourceGroup/providers/Microsoft.Storage/storageAccounts/myStorageAccount'}]}, 'sku': {'name': 'S0'}}).result()
    print(response)
if __name__ == '__main__':
    main()