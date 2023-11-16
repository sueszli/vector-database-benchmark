from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdbforpostgresql import CosmosdbForPostgresqlMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdbforpostgresql\n# USAGE\n    python configuration_update_node.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CosmosdbForPostgresqlMgmtClient(credential=DefaultAzureCredential(), subscription_id='ffffffff-ffff-ffff-ffff-ffffffffffff')
    response = client.configurations.begin_update_on_node(resource_group_name='TestResourceGroup', cluster_name='testcluster', configuration_name='array_nulls', parameters={'properties': {'value': 'off'}}).result()
    print(response)
if __name__ == '__main__':
    main()