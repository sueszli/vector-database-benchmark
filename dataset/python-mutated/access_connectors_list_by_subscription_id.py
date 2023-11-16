from azure.identity import DefaultAzureCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databricks\n# USAGE\n    python access_connectors_list_by_subscription_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AzureDatabricksManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.access_connectors.list_by_subscription()
    for item in response:
        print(item)
if __name__ == '__main__':
    main()