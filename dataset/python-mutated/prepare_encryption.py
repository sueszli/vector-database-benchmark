from azure.identity import DefaultAzureCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databricks\n# USAGE\n    python prepare_encryption.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AzureDatabricksManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.workspaces.begin_create_or_update(resource_group_name='rg', workspace_name='myWorkspace', parameters={'location': 'westus', 'properties': {'managedResourceGroupId': '/subscriptions/subid/resourceGroups/myManagedRG', 'parameters': {'prepareEncryption': {'value': True}}}}).result()
    print(response)
if __name__ == '__main__':
    main()