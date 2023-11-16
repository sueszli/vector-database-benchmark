from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datafactory\n# USAGE\n    python integration_runtimes_list_outbound_network_dependencies_endpoints.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataFactoryManagementClient(credential=DefaultAzureCredential(), subscription_id='7ad7c73b-38b8-4df3-84ee-52ff91092f61')
    response = client.integration_runtimes.list_outbound_network_dependencies_endpoints(resource_group_name='exampleResourceGroup', factory_name='exampleFactoryName', integration_runtime_name='exampleIntegrationRuntime')
    print(response)
if __name__ == '__main__':
    main()