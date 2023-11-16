from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python revisions_get.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.container_apps_revisions.get_revision(resource_group_name='rg', container_app_name='testcontainerApp0', revision_name='testcontainerApp0-pjxhsye')
    print(response)
if __name__ == '__main__':
    main()