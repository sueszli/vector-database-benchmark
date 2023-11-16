from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python dapr_components_create_or_update_secrets.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='8efdecc5-919e-44eb-b179-915dca89ebf9')
    response = client.dapr_components.create_or_update(resource_group_name='examplerg', environment_name='myenvironment', component_name='reddog', dapr_component_envelope={'properties': {'componentType': 'state.azure.cosmosdb', 'ignoreErrors': False, 'initTimeout': '50s', 'metadata': [{'name': 'url', 'value': '<COSMOS-URL>'}, {'name': 'database', 'value': 'itemsDB'}, {'name': 'collection', 'value': 'items'}, {'name': 'masterkey', 'secretRef': 'masterkey'}], 'scopes': ['container-app-1', 'container-app-2'], 'secrets': [{'name': 'masterkey', 'value': 'keyvalue'}], 'version': 'v1'}})
    print(response)
if __name__ == '__main__':
    main()