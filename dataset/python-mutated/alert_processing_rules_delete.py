from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python alert_processing_rules_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='1e3ff1c0-771a-4119-a03b-be82a51e232d')
    response = client.alert_processing_rules.delete(resource_group_name='alertscorrelationrg', alert_processing_rule_name='DailySuppression')
    print(response)
if __name__ == '__main__':
    main()