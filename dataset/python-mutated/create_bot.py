from azure.identity import DefaultAzureCredential
from azure.mgmt.botservice import AzureBotService
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-botservice\n# USAGE\n    python create_bot.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AzureBotService(credential=DefaultAzureCredential(), subscription_id='subscription-id')
    response = client.bots.create(resource_group_name='OneResourceGroupName', resource_name='samplebotname', parameters={'etag': 'etag1', 'kind': 'sdk', 'location': 'West US', 'properties': {'cmekKeyVaultUrl': 'https://myCmekKey', 'description': 'The description of the bot', 'developerAppInsightKey': 'appinsightskey', 'developerAppInsightsApiKey': 'appinsightsapikey', 'developerAppInsightsApplicationId': 'appinsightsappid', 'disableLocalAuth': True, 'displayName': 'The Name of the bot', 'endpoint': 'http://mybot.coffee', 'iconUrl': 'http://myicon', 'isCmekEnabled': True, 'luisAppIds': ['luisappid1', 'luisappid2'], 'luisKey': 'luiskey', 'msaAppId': 'exampleappid', 'msaAppMSIResourceId': '/subscriptions/foo/resourcegroups/bar/providers/microsoft.managedidentity/userassignedidentities/sampleId', 'msaAppTenantId': 'exampleapptenantid', 'msaAppType': 'UserAssignedMSI', 'publicNetworkAccess': 'Enabled', 'schemaTransformationVersion': '1.0'}, 'sku': {'name': 'S1'}, 'tags': {'tag1': 'value1', 'tag2': 'value2'}})
    print(response)
if __name__ == '__main__':
    main()