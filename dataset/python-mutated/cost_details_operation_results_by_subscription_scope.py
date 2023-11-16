from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python cost_details_operation_results_by_subscription_scope.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.generate_cost_details_report.begin_get_operation_results(scope='subscriptions/00000000-0000-0000-0000-000000000000', operation_id='00000000-0000-0000-0000-000000000000').result()
    print(response)
if __name__ == '__main__':
    main()