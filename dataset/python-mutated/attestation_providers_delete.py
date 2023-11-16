from azure.identity import DefaultAzureCredential
from azure.mgmt.attestation import AttestationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-attestation\n# USAGE\n    python attestation_providers_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AttestationManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.attestation_providers.delete(resource_group_name='sample-resource-group', provider_name='myattestationprovider')
    print(response)
if __name__ == '__main__':
    main()