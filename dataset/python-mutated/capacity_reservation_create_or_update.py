from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python capacity_reservation_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.capacity_reservations.begin_create_or_update(resource_group_name='myResourceGroup', capacity_reservation_group_name='myCapacityReservationGroup', capacity_reservation_name='myCapacityReservation', parameters={'location': 'westus', 'sku': {'capacity': 4, 'name': 'Standard_DS1_v2'}, 'tags': {'department': 'HR'}, 'zones': ['1']}).result()
    print(response)
if __name__ == '__main__':
    main()