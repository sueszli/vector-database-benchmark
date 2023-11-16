from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python origins_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.origins.begin_update(resource_group_name='RG', profile_name='profile1', endpoint_name='endpoint1', origin_name='www-someDomain-net', origin_update_properties={'properties': {'enabled': True, 'httpPort': 42, 'httpsPort': 43, 'originHostHeader': 'www.someDomain2.net', 'priority': 1, 'privateLinkAlias': 'APPSERVER.d84e61f0-0870-4d24-9746-7438fa0019d1.westus2.azure.privatelinkservice', 'weight': 50}}).result()
    print(response)
if __name__ == '__main__':
    main()