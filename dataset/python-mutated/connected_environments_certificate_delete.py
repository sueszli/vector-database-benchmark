from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python connected_environments_certificate_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    client.connected_environments_certificates.delete(resource_group_name='examplerg', connected_environment_name='testcontainerenv', certificate_name='certificate-firendly-name')
if __name__ == '__main__':
    main()