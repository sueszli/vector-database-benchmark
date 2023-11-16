from azure.identity import DefaultAzureCredential
from azure.mgmt.avs import AVSClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-avs\n# USAGE\n    python workload_networks_create_segments.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AVSClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.workload_networks.begin_create_segments(resource_group_name='group1', private_cloud_name='cloud1', segment_id='segment1', workload_network_segment={'properties': {'connectedGateway': '/infra/tier-1s/gateway', 'displayName': 'segment1', 'revision': 1, 'subnet': {'dhcpRanges': ['40.20.0.0-40.20.0.1'], 'gatewayAddress': '40.20.20.20/16'}}}).result()
    print(response)
if __name__ == '__main__':
    main()