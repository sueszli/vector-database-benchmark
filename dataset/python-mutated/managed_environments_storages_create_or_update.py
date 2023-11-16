from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python managed_environments_storages_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='8efdecc5-919e-44eb-b179-915dca89ebf9')
    response = client.managed_environments_storages.create_or_update(resource_group_name='examplerg', environment_name='managedEnv', storage_name='jlaw-demo1', storage_envelope={'properties': {'azureFile': {'accessMode': 'ReadOnly', 'accountKey': 'key', 'accountName': 'account1', 'shareName': 'share1'}}})
    print(response)
if __name__ == '__main__':
    main()