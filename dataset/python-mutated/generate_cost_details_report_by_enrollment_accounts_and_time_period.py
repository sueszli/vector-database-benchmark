from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python generate_cost_details_report_by_enrollment_accounts_and_time_period.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.generate_cost_details_report.begin_create_operation(scope='providers/Microsoft.Billing/enrollmentAccounts/1234', parameters={'metric': 'ActualCost', 'timePeriod': {'end': '2020-03-15', 'start': '2020-03-01'}}).result()
    print(response)
if __name__ == '__main__':
    main()