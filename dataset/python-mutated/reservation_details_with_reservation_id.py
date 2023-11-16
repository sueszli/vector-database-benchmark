from azure.identity import DefaultAzureCredential
from azure.mgmt.consumption import ConsumptionManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-consumption\n# USAGE\n    python reservation_details_with_reservation_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ConsumptionManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.reservations_details.list_by_reservation_order_and_reservation(reservation_order_id='00000000-0000-0000-0000-000000000000', reservation_id='00000000-0000-0000-0000-000000000000', filter='properties/usageDate ge 2017-10-01 AND properties/usageDate le 2017-12-05')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()