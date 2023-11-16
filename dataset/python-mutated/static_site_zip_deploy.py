from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python static_site_zip_deploy.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.static_sites.begin_create_zip_deployment_for_static_site(resource_group_name='rg', name='testStaticSite0', static_site_zip_deployment_envelope={'properties': {'apiZipUrl': 'https://teststorageaccount.net/happy-sea-15afae3e-master-81828877/api-zipdeploy.zip', 'appZipUrl': 'https://teststorageaccount.net/happy-sea-15afae3e-master-81828877/app-zipdeploy.zip', 'deploymentTitle': 'Update index.html', 'functionLanguage': 'testFunctionLanguage', 'provider': 'testProvider'}}).result()
    print(response)
if __name__ == '__main__':
    main()