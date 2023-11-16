from azure.identity import DefaultAzureCredential
from azure.mgmt.applicationinsights import ApplicationInsightsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-applicationinsights\n# USAGE\n    python web_test_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ApplicationInsightsManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.web_tests.delete(resource_group_name='my-resource-group', web_test_name='my-webtest-01-mywebservice')
    print(response)
if __name__ == '__main__':
    main()