from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python smart_groups_get_by_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='9e261de7-c804-4b9d-9ebf-6f50fe350a9a')
    response = client.smart_groups.get_by_id(smart_group_id='603675da-9851-4b26-854a-49fc53d32715')
    print(response)
if __name__ == '__main__':
    main()