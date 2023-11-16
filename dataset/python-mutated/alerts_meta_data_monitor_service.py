from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python alerts_meta_data_monitor_service.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.alerts.meta_data(identifier='MonitorServiceList')
    print(response)
if __name__ == '__main__':
    main()