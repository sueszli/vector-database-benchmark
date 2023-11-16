from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python profiles_migrate.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.profiles.begin_migrate(resource_group_name='RG', migration_parameters={'classicResourceReference': {'id': '/subscriptions/subid/resourcegroups/RG/providers/Microsoft.Network/frontdoors/frontdoorname'}, 'profileName': 'profile1', 'sku': {'name': 'Standard_AzureFrontDoor'}}).result()
    print(response)
if __name__ == '__main__':
    main()