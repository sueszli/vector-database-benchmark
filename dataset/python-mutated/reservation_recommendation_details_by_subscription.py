from azure.identity import DefaultAzureCredential
from azure.mgmt.consumption import ConsumptionManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-consumption\n# USAGE\n    python reservation_recommendation_details_by_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ConsumptionManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.reservation_recommendation_details.get(resource_scope='subscriptions/00000000-0000-0000-0000-00000000', scope='Single', region='westus', term='P3Y', look_back_period='Last30Days', product='Standard_DS13_v2')
    print(response)
if __name__ == '__main__':
    main()