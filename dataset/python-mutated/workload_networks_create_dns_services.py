from azure.identity import DefaultAzureCredential
from azure.mgmt.avs import AVSClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-avs\n# USAGE\n    python workload_networks_create_dns_services.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AVSClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.workload_networks.begin_create_dns_service(resource_group_name='group1', private_cloud_name='cloud1', dns_service_id='dnsService1', workload_network_dns_service={'properties': {'defaultDnsZone': 'defaultDnsZone1', 'displayName': 'dnsService1', 'dnsServiceIp': '5.5.5.5', 'fqdnZones': ['fqdnZone1'], 'logLevel': 'INFO', 'revision': 1}}).result()
    print(response)
if __name__ == '__main__':
    main()