from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python dapr_components_get_secret_store_component.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='8efdecc5-919e-44eb-b179-915dca89ebf9')
    response = client.dapr_components.get(resource_group_name='examplerg', environment_name='myenvironment', component_name='reddog')
    print(response)
if __name__ == '__main__':
    main()