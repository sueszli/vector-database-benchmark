from azure.identity import DefaultAzureCredential
from azure.mgmt.agrifood import AgriFoodMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-agrifood\n# USAGE\n    python farm_beats_extensions_get.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AgriFoodMgmtClient(credential=DefaultAzureCredential(), solution_id='SOLUTION_ID', subscription_id='SUBSCRIPTION_ID')
    response = client.farm_beats_extensions.get(farm_beats_extension_id='DTN.ContentServices')
    print(response)
if __name__ == '__main__':
    main()