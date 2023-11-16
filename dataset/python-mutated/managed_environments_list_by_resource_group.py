from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python managed_environments_list_by_resource_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='8efdecc5-919e-44eb-b179-915dca89ebf9')
    response = client.managed_environments.list_by_resource_group(resource_group_name='examplerg')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()