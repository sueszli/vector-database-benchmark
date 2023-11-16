from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python get_source_control_sync_job_streams_by_stream_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.source_control_sync_job_streams.get(resource_group_name='rg', automation_account_name='myAutomationAccount33', source_control_name='MySourceControl', source_control_sync_job_id='ce6fe3e3-9db3-4096-a6b4-82bfb4c10a2b', stream_id='b86c5c31-e9fd-4734-8764-ddd6c101e706_00636596855139029522_00000000000000000007')
    print(response)
if __name__ == '__main__':
    main()