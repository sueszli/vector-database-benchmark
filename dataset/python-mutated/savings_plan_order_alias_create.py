from azure.identity import DefaultAzureCredential
from azure.mgmt.billingbenefits import BillingBenefitsRP
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billingbenefits\n# USAGE\n    python savings_plan_order_alias_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = BillingBenefitsRP(credential=DefaultAzureCredential())
    response = client.savings_plan_order_alias.begin_create(savings_plan_order_alias_name='spAlias123', body={'properties': {'appliedScopeProperties': None, 'appliedScopeType': 'Shared', 'billingPlan': 'P1M', 'billingScopeId': '/subscriptions/30000000-0000-0000-0000-000000000000', 'commitment': {'amount': 0.001, 'currencyCode': 'USD', 'grain': 'Hourly'}, 'displayName': 'Compute_SavingsPlan_10-28-2022_16-38', 'term': 'P3Y'}, 'sku': {'name': 'Compute_Savings_Plan'}}).result()
    print(response)
if __name__ == '__main__':
    main()