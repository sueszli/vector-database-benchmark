from azure.identity import DefaultAzureCredential
from azure.mgmt.botservice import AzureBotService
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-botservice\n# USAGE\n    python get_connection.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AzureBotService(credential=DefaultAzureCredential(), subscription_id='subscription-id')
    response = client.bot_connection.list_with_secrets(resource_group_name='OneResourceGroupName', resource_name='samplebotname', connection_name='sampleConnection')
    print(response)
if __name__ == '__main__':
    main()