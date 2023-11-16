from azure.identity import DefaultAzureCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databricks\n# USAGE\n    python access_connector_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AzureDatabricksManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.access_connectors.begin_create_or_update(resource_group_name='rg', connector_name='myAccessConnector', parameters={'location': 'westus'}).result()
    print(response)
if __name__ == '__main__':
    main()