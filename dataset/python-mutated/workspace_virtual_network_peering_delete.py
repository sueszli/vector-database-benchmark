from azure.identity import DefaultAzureCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databricks\n# USAGE\n    python workspace_virtual_network_peering_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AzureDatabricksManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    client.vnet_peering.begin_delete(resource_group_name='rg', workspace_name='myWorkspace', peering_name='vNetPeering').result()
if __name__ == '__main__':
    main()