from azure.identity import DefaultAzureCredential
from azure.mgmt.datashare import DataShareManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datashare\n# USAGE\n    python share_subscriptions_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DataShareManagementClient(credential=DefaultAzureCredential(), subscription_id='12345678-1234-1234-12345678abc')
    response = client.share_subscriptions.create(resource_group_name='SampleResourceGroup', account_name='Account1', share_subscription_name='ShareSubscription1', share_subscription={'properties': {'expirationDate': '2020-08-26T22:33:24.5785265Z', 'invitationId': '12345678-1234-1234-12345678abd', 'sourceShareLocation': 'eastus2'}})
    print(response)
if __name__ == '__main__':
    main()