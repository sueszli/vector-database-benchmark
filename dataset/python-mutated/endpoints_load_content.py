from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python endpoints_load_content.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    client.endpoints.begin_load_content(resource_group_name='RG', profile_name='profile1', endpoint_name='endpoint1', content_file_paths={'contentPaths': ['/folder1']}).result()
if __name__ == '__main__':
    main()