from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python create_job_schedule.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.job_schedule.create(resource_group_name='rg', automation_account_name='ContoseAutomationAccount', job_schedule_id='0fa462ba-3aa2-4138-83ca-9ebc3bc55cdc', parameters={'properties': {'parameters': {'jobscheduletag01': 'jobschedulevalue01', 'jobscheduletag02': 'jobschedulevalue02'}, 'runbook': {'name': 'TestRunbook'}, 'schedule': {'name': 'ScheduleNameGoesHere332204b5-debe-4348-a5c7-6357457189f2'}}})
    print(response)
if __name__ == '__main__':
    main()