from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python savings_plan_savings_plan_id_monthly.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.benefit_utilization_summaries.list_by_savings_plan_id(savings_plan_order_id='66cccc66-6ccc-6c66-666c-66cc6c6c66c6', savings_plan_id='222d22dd-d2d2-2dd2-222d-2dd2222ddddd')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()