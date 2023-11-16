from azure.identity import DefaultAzureCredential
from azure.mgmt.automanage import AutomanageClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automanage\n# USAGE\n    python delete_configuration_profile_version.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AutomanageClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.configuration_profiles_versions.delete(resource_group_name='rg', configuration_profile_name='customConfigurationProfile', version_name='version1')
    print(response)
if __name__ == '__main__':
    main()