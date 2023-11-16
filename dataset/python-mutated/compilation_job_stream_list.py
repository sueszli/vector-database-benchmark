from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python compilation_job_stream_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.dsc_compilation_job_stream.list_by_job(resource_group_name='rg', automation_account_name='myAutomationAccount33', job_id='836d4e06-2d88-46b4-8500-7febd4906838')
    print(response)
if __name__ == '__main__':
    main()