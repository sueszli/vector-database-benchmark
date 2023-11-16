from azure.identity import DefaultAzureCredential
from azure.mgmt.billingbenefits import BillingBenefitsRP
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billingbenefits\n# USAGE\n    python savings_plan_item_get.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = BillingBenefitsRP(credential=DefaultAzureCredential())
    response = client.savings_plan.get(savings_plan_order_id='20000000-0000-0000-0000-000000000000', savings_plan_id='30000000-0000-0000-0000-000000000000')
    print(response)
if __name__ == '__main__':
    main()