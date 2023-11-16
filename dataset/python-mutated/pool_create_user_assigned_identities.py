from azure.identity import DefaultAzureCredential
from azure.mgmt.batch import BatchManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-batch\n# USAGE\n    python pool_create_user_assigned_identities.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = BatchManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.pool.create(resource_group_name='default-azurebatch-japaneast', account_name='sampleacct', pool_name='testpool', parameters={'identity': {'type': 'UserAssigned', 'userAssignedIdentities': {'/subscriptions/subid/resourceGroups/default-azurebatch-japaneast/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id1': {}, '/subscriptions/subid/resourceGroups/default-azurebatch-japaneast/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id2': {}}}, 'properties': {'deploymentConfiguration': {'virtualMachineConfiguration': {'imageReference': {'offer': 'UbuntuServer', 'publisher': 'Canonical', 'sku': '18.04-LTS', 'version': 'latest'}, 'nodeAgentSkuId': 'batch.node.ubuntu 18.04'}}, 'scaleSettings': {'autoScale': {'evaluationInterval': 'PT5M', 'formula': '$TargetDedicatedNodes=1'}}, 'vmSize': 'STANDARD_D4'}})
    print(response)
if __name__ == '__main__':
    main()