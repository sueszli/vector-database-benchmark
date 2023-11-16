from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python create_or_update_app_service_plan.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.app_service_plans.begin_create_or_update(resource_group_name='testrg123', name='testsf6141', app_service_plan={'kind': 'app', 'location': 'East US', 'properties': {}, 'sku': {'capacity': 1, 'family': 'P', 'name': 'P1', 'size': 'P1', 'tier': 'Premium'}}).result()
    print(response)
if __name__ == '__main__':
    main()