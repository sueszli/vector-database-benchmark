from azure.identity import DefaultAzureCredential
from azure.mgmt.attestation import AttestationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-attestation\n# USAGE\n    python attestation_providers_list_by_resource_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AttestationManagementClient(credential=DefaultAzureCredential(), subscription_id='6c96b33e-f5b8-40a6-9011-5cb1c58b0915')
    response = client.attestation_providers.list_by_resource_group(resource_group_name='testrg1')
    print(response)
if __name__ == '__main__':
    main()