from azure.identity import DefaultAzureCredential
from azure.mgmt.datalake.store import DataLakeStoreAccountManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datalake-store\n# USAGE\n    python accounts_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataLakeStoreAccountManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.accounts.begin_create(resource_group_name='contosorg', account_name='contosoadla', parameters={'identity': {'type': 'SystemAssigned'}, 'location': 'eastus2', 'properties': {'defaultGroup': 'test_default_group', 'encryptionConfig': {'keyVaultMetaInfo': {'encryptionKeyName': 'test_encryption_key_name', 'encryptionKeyVersion': 'encryption_key_version', 'keyVaultResourceId': '34adfa4f-cedf-4dc0-ba29-b6d1a69ab345'}, 'type': 'UserManaged'}, 'encryptionState': 'Enabled', 'firewallAllowAzureIps': 'Enabled', 'firewallRules': [{'name': 'test_rule', 'properties': {'endIpAddress': '2.2.2.2', 'startIpAddress': '1.1.1.1'}}], 'firewallState': 'Enabled', 'newTier': 'Consumption', 'trustedIdProviderState': 'Enabled', 'trustedIdProviders': [{'name': 'test_trusted_id_provider_name', 'properties': {'idProvider': 'https://sts.windows.net/ea9ec534-a3e3-4e45-ad36-3afc5bb291c1'}}]}, 'tags': {'test_key': 'test_value'}}).result()
    print(response)
if __name__ == '__main__':
    main()