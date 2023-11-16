from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerinstance\n# USAGE\n    python container_groups_list_by_resource_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerInstanceManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.container_groups.list_by_resource_group(resource_group_name='demo')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()