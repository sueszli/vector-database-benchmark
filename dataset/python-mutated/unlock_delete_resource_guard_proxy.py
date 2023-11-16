from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python unlock_delete_resource_guard_proxy.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='5e13b949-1218-4d18-8b99-7e12155ec4f7')
    response = client.dpp_resource_guard_proxy.unlock_delete(resource_group_name='SampleResourceGroup', vault_name='sampleVault', resource_guard_proxy_name='swaggerExample', parameters={'resourceGuardOperationRequests': ['/subscriptions/f9e67185-f313-4e79-aa71-6458d429369d/resourceGroups/ResourceGuardSecurityAdminRG/providers/Microsoft.DataProtection/resourceGuards/ResourceGuardTestResource/deleteBackupInstanceRequests/default'], 'resourceToBeDeleted': '/subscriptions/5e13b949-1218-4d18-8b99-7e12155ec4f7/resourceGroups/SampleResourceGroup/providers/Microsoft.DataProtection/backupVaults/sampleVault/backupInstances/TestBI9779f4de'})
    print(response)
if __name__ == '__main__':
    main()