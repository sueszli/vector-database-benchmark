from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python afd_origin_groups_list_by_profile.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.afd_origin_groups.list_by_profile(resource_group_name='RG', profile_name='profile1')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()