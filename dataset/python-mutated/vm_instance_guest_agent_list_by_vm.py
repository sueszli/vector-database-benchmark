from azure.identity import DefaultAzureCredential
from azure.mgmt.connectedvmware import ConnectedVMwareMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-connectedvmware\n# USAGE\n    python vm_instance_guest_agent_list_by_vm.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ConnectedVMwareMgmtClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.vm_instance_guest_agents.list(resource_uri='subscriptions/fd3c3665-1729-4b7b-9a38-238e83b0f98b/resourceGroups/testrg/providers/Microsoft.HybridCompute/machines/DemoVM')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()