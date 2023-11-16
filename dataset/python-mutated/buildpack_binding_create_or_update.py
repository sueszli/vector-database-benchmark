from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python buildpack_binding_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.buildpack_binding.begin_create_or_update(resource_group_name='myResourceGroup', service_name='myservice', build_service_name='default', builder_name='default', buildpack_binding_name='myBuildpackBinding', buildpack_binding={'properties': {'bindingType': 'ApplicationInsights', 'launchProperties': {'properties': {'abc': 'def', 'any-string': 'any-string', 'sampling-rate': '12.0'}, 'secrets': {'connection-string': 'XXXXXXXXXXXXXXXXX=XXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXX;XXXXXXXXXXXXXXXXX=XXXXXXXXXXXXXXXXXXX'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()