from azure.identity import DefaultAzureCredential
from azure.mgmt.automanage import AutomanageClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automanage\n# USAGE\n    python list_best_practices_versions_by_tenant.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AutomanageClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.best_practices_versions.list_by_tenant(best_practice_name='azureBestPracticesProduction')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()