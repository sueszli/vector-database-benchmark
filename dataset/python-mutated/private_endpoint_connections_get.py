from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservice\n# USAGE\n    python private_endpoint_connections_get.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ContainerServiceClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.private_endpoint_connections.get(resource_group_name='rg1', resource_name='clustername1', private_endpoint_connection_name='privateendpointconnection1')
    print(response)
if __name__ == '__main__':
    main()