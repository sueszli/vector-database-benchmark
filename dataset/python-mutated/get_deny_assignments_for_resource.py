from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-authorization\n# USAGE\n    python get_deny_assignments_for_resource.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AuthorizationManagementClient(credential=DefaultAzureCredential(), subscription_id='subId')
    response = client.deny_assignments.list_for_resource(resource_group_name='rgname', resource_provider_namespace='resourceProviderNamespace', parent_resource_path='parentResourcePath', resource_type='resourceType', resource_name='resourceName')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()