from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python alerts_history.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='9e261de7-c804-4b9d-9ebf-6f50fe350a9a')
    response = client.alerts.get_history(alert_id='66114d64-d9d9-478b-95c9-b789d6502100')
    print(response)
if __name__ == '__main__':
    main()