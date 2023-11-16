from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python clone_web_app.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.web_apps.begin_create_or_update(resource_group_name='testrg123', name='sitef6141', site_envelope={'kind': 'app', 'location': 'East US', 'properties': {'cloningInfo': {'appSettingsOverrides': {'Setting1': 'NewValue1', 'Setting3': 'NewValue5'}, 'cloneCustomHostNames': True, 'cloneSourceControl': True, 'configureLoadBalancing': False, 'hostingEnvironment': '/subscriptions/34adfa4f-cedf-4dc0-ba29-b6d1a69ab345/resourceGroups/testrg456/providers/Microsoft.Web/hostingenvironments/aseforsites', 'overwrite': False, 'sourceWebAppId': '/subscriptions/34adfa4f-cedf-4dc0-ba29-b6d1a69ab345/resourceGroups/testrg456/providers/Microsoft.Web/sites/srcsiteg478', 'sourceWebAppLocation': 'West Europe'}}}).result()
    print(response)
if __name__ == '__main__':
    main()