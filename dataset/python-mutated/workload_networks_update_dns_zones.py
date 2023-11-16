from azure.identity import DefaultAzureCredential
from azure.mgmt.avs import AVSClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-avs\n# USAGE\n    python workload_networks_update_dns_zones.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AVSClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.workload_networks.begin_update_dns_zone(resource_group_name='group1', private_cloud_name='cloud1', dns_zone_id='dnsZone1', workload_network_dns_zone={'properties': {'displayName': 'dnsZone1', 'dnsServerIps': ['1.1.1.1'], 'domain': [], 'revision': 1, 'sourceIp': '8.8.8.8'}}).result()
    print(response)
if __name__ == '__main__':
    main()