from azure.identity import DefaultAzureCredential
from azure.mgmt.attestation import AttestationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-attestation\n# USAGE\n    python attestation_provider_list_private_endpoint_connections.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AttestationManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.private_endpoint_connections.list(resource_group_name='res6977', provider_name='sto2527')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()