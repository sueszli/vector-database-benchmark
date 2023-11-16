from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python get_test_job_stream.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='51766542-3ed7-4a72-a187-0c8ab644ddab')
    response = client.test_job_streams.get(resource_group_name='mygroup', automation_account_name='ContoseAutomationAccount', runbook_name='Get-AzureVMTutorial', job_stream_id='851b2101-686f-40e2-8a4b-5b8df08afbd1_00636535684910693884_00000000000000000001')
    print(response)
if __name__ == '__main__':
    main()