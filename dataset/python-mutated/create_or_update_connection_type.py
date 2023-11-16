from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python create_or_update_connection_type.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.connection_type.create_or_update(resource_group_name='rg', automation_account_name='myAutomationAccount22', connection_type_name='myCT', parameters={'name': 'myCT', 'properties': {'fieldDefinitions': {'myBoolField': {'isEncrypted': False, 'isOptional': False, 'type': 'bool'}, 'myStringField': {'isEncrypted': False, 'isOptional': False, 'type': 'string'}, 'myStringFieldEncrypted': {'isEncrypted': True, 'isOptional': False, 'type': 'string'}}, 'isGlobal': False}})
    print(response)
if __name__ == '__main__':
    main()