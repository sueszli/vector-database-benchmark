from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python certificate_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.certificates.create_or_update(resource_group_name='examplerg', environment_name='testcontainerenv', certificate_name='certificate-firendly-name')
    print(response)
if __name__ == '__main__':
    main()