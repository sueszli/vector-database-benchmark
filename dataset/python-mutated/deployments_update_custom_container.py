from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python deployments_update_custom_container.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.deployments.begin_update(resource_group_name='myResourceGroup', service_name='myservice', app_name='myapp', deployment_name='mydeployment', deployment_resource={'properties': {'instances': None, 'source': {'customContainer': {'args': ['-c', 'while true; do echo hello; sleep 10;done'], 'command': ['/bin/sh'], 'containerImage': 'myNewContainerImage:v1', 'imageRegistryCredential': {'password': '<myNewPassword>', 'username': 'myNewUsername'}, 'server': 'mynewacr.azurecr.io'}, 'type': 'Container'}}}).result()
    print(response)
if __name__ == '__main__':
    main()