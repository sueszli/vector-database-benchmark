from azure.identity import DefaultAzureCredential
from azure.mgmt.defendereasm import EasmMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-defendereasm\n# USAGE\n    python labels_create_and_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = EasmMgmtClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.labels.begin_create_and_update(resource_group_name='dummyrg', workspace_name='ThisisaWorkspace', label_name='ThisisaLabel').result()
    print(response)
if __name__ == '__main__':
    main()