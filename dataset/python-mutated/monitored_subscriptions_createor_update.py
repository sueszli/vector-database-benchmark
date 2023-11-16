from azure.identity import DefaultAzureCredential
from azure.mgmt.datadog import MicrosoftDatadogClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datadog\n# USAGE\n    python monitored_subscriptions_createor_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = MicrosoftDatadogClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.monitored_subscriptions.begin_createor_update(resource_group_name='myResourceGroup', monitor_name='myMonitor', configuration_name='default').result()
    print(response)
if __name__ == '__main__':
    main()