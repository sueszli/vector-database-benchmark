from azure.identity import DefaultAzureCredential
from azure.mgmt.billingbenefits import BillingBenefitsRP
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billingbenefits\n# USAGE\n    python savings_plan_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = BillingBenefitsRP(credential=DefaultAzureCredential())
    response = client.savings_plan.update(savings_plan_order_id='20000000-0000-0000-0000-000000000000', savings_plan_id='30000000-0000-0000-0000-000000000000', body={'properties': {'appliedScopeProperties': {'resourceGroupId': '/subscriptions/10000000-0000-0000-0000-000000000000/resourceGroups/testrg'}, 'appliedScopeType': 'Single', 'displayName': 'TestDisplayName', 'renew': True, 'renewProperties': {'purchaseProperties': {'properties': {'appliedScopeProperties': {'resourceGroupId': '/subscriptions/10000000-0000-0000-0000-000000000000/resourceGroups/testrg'}, 'appliedScopeType': 'Single', 'billingPlan': 'P1M', 'billingScopeId': '/subscriptions/10000000-0000-0000-0000-000000000000', 'commitment': {'amount': 15.23, 'currencyCode': 'USD', 'grain': 'Hourly'}, 'displayName': 'TestDisplayName_renewed', 'renew': False, 'term': 'P1Y'}, 'sku': {'name': 'Compute_Savings_Plan'}}}}})
    print(response)
if __name__ == '__main__':
    main()