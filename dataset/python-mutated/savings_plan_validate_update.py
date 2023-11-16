from azure.identity import DefaultAzureCredential
from azure.mgmt.billingbenefits import BillingBenefitsRP
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billingbenefits\n# USAGE\n    python savings_plan_validate_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = BillingBenefitsRP(credential=DefaultAzureCredential())
    response = client.savings_plan.validate_update(savings_plan_order_id='20000000-0000-0000-0000-000000000000', savings_plan_id='30000000-0000-0000-0000-000000000000', body={'benefits': [{'appliedScopeProperties': {'managementGroupId': '/providers/Microsoft.Management/managementGroups/30000000-0000-0000-0000-000000000100', 'tenantId': '30000000-0000-0000-0000-000000000100'}, 'appliedScopeType': 'ManagementGroup'}, {'appliedScopeProperties': {'managementGroupId': '/providers/Microsoft.Management/managementGroups/MockMG', 'tenantId': '30000000-0000-0000-0000-000000000100'}, 'appliedScopeType': 'ManagementGroup'}]})
    print(response)
if __name__ == '__main__':
    main()