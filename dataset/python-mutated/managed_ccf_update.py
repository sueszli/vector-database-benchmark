from azure.identity import DefaultAzureCredential
from azure.mgmt.confidentialledger import ConfidentialLedger
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-confidentialledger\n# USAGE\n    python managed_ccf_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ConfidentialLedger(credential=DefaultAzureCredential(), subscription_id='0000000-0000-0000-0000-000000000001')
    client.managed_ccf.begin_update(resource_group_name='DummyResourceGroupName', app_name='DummyMccfAppName', managed_ccf={'location': 'EastUS', 'properties': {'deploymentType': {'appSourceUri': 'https://myaccount.blob.core.windows.net/storage/mccfsource?sv=2022-02-11%st=2022-03-11', 'languageRuntime': 'CPP'}}, 'tags': {'additionalProps1': 'additional properties'}}).result()
if __name__ == '__main__':
    main()