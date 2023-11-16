from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservicefleet import ContainerServiceFleetMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservicefleet\n# USAGE\n    python fleet_members_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ContainerServiceFleetMgmtClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.fleet_members.begin_create(resource_group_name='rg1', fleet_name='fleet1', fleet_member_name='member-1', resource={'properties': {'clusterResourceId': '/subscriptions/subid1/resourcegroups/rg1/providers/Microsoft.ContainerService/managedClusters/cluster-1'}}).result()
    print(response)
if __name__ == '__main__':
    main()