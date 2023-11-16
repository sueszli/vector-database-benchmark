from azure.identity import DefaultAzureCredential
from azure.mgmt.batch import BatchManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-batch\n# USAGE\n    python pool_create_accelerated_networking.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = BatchManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.pool.create(resource_group_name='default-azurebatch-japaneast', account_name='sampleacct', pool_name='testpool', parameters={'properties': {'deploymentConfiguration': {'virtualMachineConfiguration': {'imageReference': {'offer': 'WindowsServer', 'publisher': 'MicrosoftWindowsServer', 'sku': '2016-datacenter-smalldisk', 'version': 'latest'}, 'nodeAgentSkuId': 'batch.node.windows amd64'}}, 'networkConfiguration': {'enableAcceleratedNetworking': True, 'subnetId': '/subscriptions/subid/resourceGroups/rg1234/providers/Microsoft.Network/virtualNetworks/network1234/subnets/subnet123'}, 'scaleSettings': {'fixedScale': {'targetDedicatedNodes': 1, 'targetLowPriorityNodes': 0}}, 'vmSize': 'STANDARD_D1_V2'}})
    print(response)
if __name__ == '__main__':
    main()