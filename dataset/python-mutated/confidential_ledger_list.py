from azure.identity import DefaultAzureCredential
from azure.mgmt.confidentialledger import ConfidentialLedger
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-confidentialledger\n# USAGE\n    python confidential_ledger_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ConfidentialLedger(credential=DefaultAzureCredential(), subscription_id='0000000-0000-0000-0000-000000000001')
    response = client.ledger.list_by_resource_group(resource_group_name='DummyResourceGroupName')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()