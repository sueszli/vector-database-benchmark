from azure.identity import DefaultAzureCredential
from azure.mgmt.communication import CommunicationServiceManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-communication\n# USAGE\n    python create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CommunicationServiceManagementClient(credential=DefaultAzureCredential(), subscription_id='11112222-3333-4444-5555-666677778888')
    response = client.communication_services.begin_create_or_update(resource_group_name='MyResourceGroup', communication_service_name='MyCommunicationResource', parameters={'location': 'Global', 'properties': {'dataLocation': 'United States'}}).result()
    print(response)
if __name__ == '__main__':
    main()