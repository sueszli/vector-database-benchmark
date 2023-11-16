from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python scheduled_actioncreate_or_updateshared.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.scheduled_actions.create_or_update_by_scope(scope='subscriptions/00000000-0000-0000-0000-000000000000', name='monthlyCostByResource', scheduled_action={'kind': 'Email', 'properties': {'displayName': 'Monthly Cost By Resource', 'fileDestination': {'fileFormats': ['Csv']}, 'notification': {'subject': 'Cost by resource this month', 'to': ['user@gmail.com', 'team@gmail.com']}, 'schedule': {'daysOfWeek': ['Monday'], 'endDate': '2021-06-19T22:21:51.1287144Z', 'frequency': 'Monthly', 'hourOfDay': 10, 'startDate': '2020-06-19T22:21:51.1287144Z', 'weeksOfMonth': ['First', 'Third']}, 'status': 'Enabled', 'viewId': '/providers/Microsoft.CostManagement/views/swaggerExample'}})
    print(response)
if __name__ == '__main__':
    main()