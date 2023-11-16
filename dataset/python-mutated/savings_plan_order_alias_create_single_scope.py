from azure.identity import DefaultAzureCredential
from azure.mgmt.billingbenefits import BillingBenefitsRP
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billingbenefits\n# USAGE\n    python savings_plan_order_alias_create_single_scope.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = BillingBenefitsRP(credential=DefaultAzureCredential())
    response = client.savings_plan_order_alias.begin_create(savings_plan_order_alias_name='spAlias123', body={'properties': {'appliedScopeProperties': {'subscriptionId': '/subscriptions/30000000-0000-0000-0000-000000000000'}, 'appliedScopeType': 'Single', 'billingPlan': 'P1M', 'billingScopeId': '/providers/Microsoft.Billing/billingAccounts/1234567/billingSubscriptions/30000000-0000-0000-0000-000000000000', 'commitment': {'amount': 0.001, 'currencyCode': 'USD', 'grain': 'Hourly'}, 'displayName': 'Compute_SavingsPlan_10-28-2022_16-38', 'term': 'P3Y'}, 'sku': {'name': 'Compute_Savings_Plan'}}).result()
    print(response)
if __name__ == '__main__':
    main()