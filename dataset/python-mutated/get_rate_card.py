from azure.identity import DefaultAzureCredential
from azure.mgmt.commerce import UsageManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-commerce\n# USAGE\n    python get_rate_card.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = UsageManagementClient(credential=DefaultAzureCredential(), subscription_id='6d61cc05-8f8f-4916-b1b9-f1d9c25aae27')
    response = client.rate_card.get(filter="OfferDurableId eq 'MS-AZR-0003P' and Currency eq 'USD' and Locale eq 'en-US' and RegionInfo eq 'US'")
    print(response)
if __name__ == '__main__':
    main()