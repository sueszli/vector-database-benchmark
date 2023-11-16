from azure.identity import DefaultAzureCredential
from azure.mgmt.avs import AVSClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-avs\n# USAGE\n    python workload_networks_delete_vm_groups.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AVSClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    client.workload_networks.begin_delete_vm_group(resource_group_name='group1', vm_group_id='vmGroup1', private_cloud_name='cloud1').result()
if __name__ == '__main__':
    main()