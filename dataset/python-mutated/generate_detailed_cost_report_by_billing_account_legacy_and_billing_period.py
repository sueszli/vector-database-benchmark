from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python generate_detailed_cost_report_by_billing_account_legacy_and_billing_period.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.generate_detailed_cost_report.begin_create_operation(scope='providers/Microsoft.Billing/billingAccounts/12345', parameters={'billingPeriod': '202008', 'metric': 'ActualCost'}).result()
    print(response)
if __name__ == '__main__':
    main()