from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservice\n# USAGE\n    python agent_pools_get_upgrade_profile.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerServiceClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.agent_pools.get_upgrade_profile(resource_group_name='rg1', resource_name='clustername1', agent_pool_name='agentpool1')
    print(response)
if __name__ == '__main__':
    main()