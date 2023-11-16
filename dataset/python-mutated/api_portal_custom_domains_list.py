from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python api_portal_custom_domains_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.api_portal_custom_domains.list(resource_group_name='myResourceGroup', service_name='myservice', api_portal_name='default')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()