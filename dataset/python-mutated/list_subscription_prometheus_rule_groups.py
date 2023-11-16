from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python list_subscription_prometheus_rule_groups.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='14ddf0c5-77c5-4b53-84f6-e1fa43ad68f7')
    response = client.prometheus_rule_groups.list_by_subscription()
    for item in response:
        print(item)
if __name__ == '__main__':
    main()