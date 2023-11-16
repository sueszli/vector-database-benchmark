from azure.identity import DefaultAzureCredential
from azure.mgmt.datalake.analytics import DataLakeAnalyticsAccountManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datalake-analytics\n# USAGE\n    python storage_accounts_add.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataLakeAnalyticsAccountManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.storage_accounts.add(resource_group_name='contosorg', account_name='contosoadla', storage_account_name='test_storage', parameters={'properties': {'accessKey': '34adfa4f-cedf-4dc0-ba29-b6d1a69ab346', 'suffix': 'test_suffix'}})
    print(response)
if __name__ == '__main__':
    main()