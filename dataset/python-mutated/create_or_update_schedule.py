from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python create_or_update_schedule.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.schedule.create_or_update(resource_group_name='rg', automation_account_name='myAutomationAccount33', schedule_name='mySchedule', parameters={'name': 'mySchedule', 'properties': {'advancedSchedule': {}, 'description': 'my description of schedule goes here', 'expiryTime': '2017-04-01T17:28:57.2494819Z', 'frequency': 'Hour', 'interval': 1, 'startTime': '2017-03-27T17:28:57.2494819Z'}})
    print(response)
if __name__ == '__main__':
    main()