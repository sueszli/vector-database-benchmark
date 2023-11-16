from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python dedicated_host_group_update_maximum_set_gen.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.dedicated_host_groups.update(resource_group_name='rgcompute', host_group_name='aaaa', parameters={'properties': {'instanceView': {'hosts': [{'availableCapacity': {'allocatableVMs': [{'count': 26, 'vmSize': 'aaaaaaaaaaaaaaaaaaaa'}]}, 'statuses': [{'code': 'aaaaaaaaaaaaaaaaaaaaaaa', 'displayStatus': 'aaaaaa', 'level': 'Info', 'message': 'a', 'time': '2021-11-30T12:58:26.522Z'}]}]}, 'platformFaultDomainCount': 3, 'supportAutomaticPlacement': True}, 'tags': {'key9921': 'aaaaaaaaaa'}, 'zones': ['aaaaaaaaaaaaaaaaaaaaaaaaaaaaa']})
    print(response)
if __name__ == '__main__':
    main()