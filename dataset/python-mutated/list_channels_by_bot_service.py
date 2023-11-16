from azure.identity import DefaultAzureCredential
from azure.mgmt.botservice import AzureBotService
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-botservice\n# USAGE\n    python list_channels_by_bot_service.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AzureBotService(credential=DefaultAzureCredential(), subscription_id='subscription-id')
    response = client.channels.list_by_resource_group(resource_group_name='OneResourceGroupName', resource_name='samplebotname')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()