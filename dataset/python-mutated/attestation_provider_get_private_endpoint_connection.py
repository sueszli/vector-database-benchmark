from azure.identity import DefaultAzureCredential
from azure.mgmt.attestation import AttestationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-attestation\n# USAGE\n    python attestation_provider_get_private_endpoint_connection.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AttestationManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.private_endpoint_connections.get(resource_group_name='res6977', provider_name='sto2527', private_endpoint_connection_name='{privateEndpointConnectionName}')
    print(response)
if __name__ == '__main__':
    main()