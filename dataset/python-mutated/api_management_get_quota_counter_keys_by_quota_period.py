from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_get_quota_counter_keys_by_quota_period.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.quota_by_period_keys.get(resource_group_name='rg1', service_name='apimService1', quota_counter_key='ba', quota_period_key='0_P3Y6M4DT12H30M5S')
    print(response)
if __name__ == '__main__':
    main()