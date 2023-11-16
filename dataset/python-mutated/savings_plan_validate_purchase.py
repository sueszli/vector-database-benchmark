from azure.identity import DefaultAzureCredential
from azure.mgmt.billingbenefits import BillingBenefitsRP
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billingbenefits\n# USAGE\n    python savings_plan_validate_purchase.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = BillingBenefitsRP(credential=DefaultAzureCredential())
    response = client.validate_purchase(body={'benefits': [{'properties': {'appliedScopeProperties': {'resourceGroupId': '/subscriptions/10000000-0000-0000-0000-000000000000/resourceGroups/testrg'}, 'appliedScopeType': 'Single', 'billingScopeId': '/subscriptions/10000000-0000-0000-0000-000000000000', 'commitment': {'amount': 15.23, 'currencyCode': 'USD', 'grain': 'Hourly'}, 'displayName': 'ComputeSavingsPlan_2021-07-01', 'term': 'P1Y'}, 'sku': {'name': 'Compute_Savings_Plan'}}, {'properties': {'appliedScopeProperties': {'resourceGroupId': '/subscriptions/10000000-0000-0000-0000-000000000000/resourceGroups/RG'}, 'appliedScopeType': 'Single', 'billingScopeId': '/subscriptions/10000000-0000-0000-0000-000000000000', 'commitment': {'amount': 20, 'currencyCode': 'USD', 'grain': 'Hourly'}, 'displayName': 'ComputeSavingsPlan_2021-07-01', 'term': 'P1Y'}, 'sku': {'name': 'Compute_Savings_Plan'}}]})
    print(response)
if __name__ == '__main__':
    main()