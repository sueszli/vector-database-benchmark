from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python update_auth_settings.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.web_apps.update_auth_settings(resource_group_name='testrg123', name='sitef6141', site_auth_settings={'properties': {'allowedExternalRedirectUrls': ['sitef6141.customdomain.net', 'sitef6141.customdomain.info'], 'clientId': '42d795a9-8abb-4d06-8534-39528af40f8e.apps.googleusercontent.com', 'defaultProvider': 'Google', 'enabled': True, 'runtimeVersion': '~1', 'tokenRefreshExtensionHours': 120, 'tokenStoreEnabled': True, 'unauthenticatedClientAction': 'RedirectToLoginPage'}})
    print(response)
if __name__ == '__main__':
    main()