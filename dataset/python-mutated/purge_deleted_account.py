from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cognitiveservices\n# USAGE\n    python purge_deleted_account.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CognitiveServicesManagementClient(credential=DefaultAzureCredential(), subscription_id='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')
    client.deleted_accounts.begin_purge(location='westus', resource_group_name='myResourceGroup', account_name='PropTest01').result()
if __name__ == '__main__':
    main()