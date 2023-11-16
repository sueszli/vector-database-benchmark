from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python capacity_reservation_update_maximum_set_gen.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.capacity_reservations.begin_update(resource_group_name='rgcompute', capacity_reservation_group_name='aaaaaaaaaa', capacity_reservation_name='aaaaaaaaaaaaaaaaaaa', parameters={'properties': {'instanceView': {'statuses': [{'code': 'aaaaaaaaaaaaaaaaaaaaaaa', 'displayStatus': 'aaaaaa', 'level': 'Info', 'message': 'a', 'time': '2021-11-30T12:58:26.522Z'}], 'utilizationInfo': {}}}, 'sku': {'capacity': 7, 'name': 'Standard_DS1_v2', 'tier': 'aaa'}, 'tags': {'key4974': 'aaaaaaaaaaaaaaaa'}}).result()
    print(response)
if __name__ == '__main__':
    main()