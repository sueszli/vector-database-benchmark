from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerinstance\n# USAGE\n    python subnet_service_association_link_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ContainerInstanceManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.subnet_service_association_link.begin_delete(resource_group_name='demo', virtual_network_name='demo2', subnet_name='demo3').result()
    print(response)
if __name__ == '__main__':
    main()