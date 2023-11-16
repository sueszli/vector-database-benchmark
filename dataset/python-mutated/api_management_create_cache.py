from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_cache.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.cache.create_or_update(resource_group_name='rg1', service_name='apimService1', cache_id='c1', parameters={'properties': {'connectionString': 'apim.redis.cache.windows.net:6380,password=xc,ssl=True,abortConnect=False', 'description': 'Redis cache instances in West India', 'resourceId': 'https://management.azure.com/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.Cache/redis/apimservice1', 'useFromLocation': 'default'}})
    print(response)
if __name__ == '__main__':
    main()