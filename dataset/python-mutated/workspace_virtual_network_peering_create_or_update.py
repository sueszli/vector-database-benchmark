from azure.identity import DefaultAzureCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databricks\n# USAGE\n    python workspace_virtual_network_peering_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AzureDatabricksManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.vnet_peering.begin_create_or_update(resource_group_name='rg', workspace_name='myWorkspace', peering_name='vNetPeeringTest', virtual_network_peering_parameters={'properties': {'allowForwardedTraffic': False, 'allowGatewayTransit': False, 'allowVirtualNetworkAccess': True, 'remoteVirtualNetwork': {'id': '/subscriptions/0140911e-1040-48da-8bc9-b99fb3dd88a6/resourceGroups/subramantest/providers/Microsoft.Network/virtualNetworks/subramanvnet'}, 'useRemoteGateways': False}}).result()
    print(response)
if __name__ == '__main__':
    main()