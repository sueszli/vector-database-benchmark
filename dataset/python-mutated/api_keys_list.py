from azure.identity import DefaultAzureCredential
from azure.mgmt.datadog import MicrosoftDatadogClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datadog\n# USAGE\n    python api_keys_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = MicrosoftDatadogClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.monitors.list_api_keys(resource_group_name='myResourceGroup', monitor_name='myMonitor')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()