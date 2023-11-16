from azure.identity import DefaultAzureCredential
from azure.mgmt.consumption import ConsumptionManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-consumption\n# USAGE\n    python reservation_transactions_list_by_billing_profile_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ConsumptionManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.reservation_transactions.list_by_billing_profile(billing_account_id='fcebaabc-fced-4284-a83d-79f83dee183c:45796ba8-988f-45ad-bea9-7b71fc6c7513_2018-09-30', billing_profile_id='Z76D-SGAF-BG7-TGB')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()