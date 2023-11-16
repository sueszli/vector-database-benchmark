from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python apps_update_vnet_injection.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.apps.begin_update(resource_group_name='myResourceGroup', service_name='myservice', app_name='myapp', app_resource={'identity': {'principalId': None, 'tenantId': None, 'type': 'SystemAssigned,UserAssigned', 'userAssignedIdentities': {'/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/samplegroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id1': {'clientId': None, 'principalId': None}, '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/samplegroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id2': {'clientId': None, 'principalId': None}}}, 'location': 'eastus', 'properties': {'customPersistentDisks': [{'customPersistentDiskProperties': {'mountOptions': [], 'mountPath': '/mypath1/mypath2', 'shareName': 'myFileShare', 'type': 'AzureFileVolume'}, 'storageId': 'myASCStorageID'}], 'enableEndToEndTLS': False, 'httpsOnly': False, 'persistentDisk': {'mountPath': '/mypersistentdisk', 'sizeInGB': 2}, 'public': True, 'temporaryDisk': {'mountPath': '/mytemporarydisk', 'sizeInGB': 2}, 'vnetAddons': {'publicEndpoint': True}}}).result()
    print(response)
if __name__ == '__main__':
    main()