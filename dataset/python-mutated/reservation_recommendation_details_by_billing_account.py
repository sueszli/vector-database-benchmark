from azure.identity import DefaultAzureCredential
from azure.mgmt.consumption import ConsumptionManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-consumption\n# USAGE\n    python reservation_recommendation_details_by_billing_account.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ConsumptionManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.reservation_recommendation_details.get(resource_scope='providers/Microsoft.Billing/billingAccounts/000000', scope='Shared', region='eastus', term='P1Y', look_back_period='Last60Days', product='Standard_DS14_v2')
    print(response)
if __name__ == '__main__':
    main()