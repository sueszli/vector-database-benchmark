from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python workflow_run_action_repetitions_request_histories_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.workflow_run_action_repetitions_request_histories.list(resource_group_name='test-resource-group', name='test-name', workflow_name='test-workflow', run_name='08586776228332053161046300351', action_name='HTTP_Webhook', repetition_name='000001')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()