from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python create_or_update_dsc_configuration.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.dsc_configuration.create_or_update(resource_group_name='rg', automation_account_name='myAutomationAccount18', configuration_name='SetupServer', parameters={'location': 'East US 2', 'name': 'SetupServer', 'properties': {'description': 'sample configuration', 'source': {'hash': {'algorithm': 'sha256', 'value': 'A9E5DB56BA21513F61E0B3868816FDC6D4DF5131F5617D7FF0D769674BD5072F'}, 'type': 'embeddedContent', 'value': 'Configuration SetupServer {\r\n    Node localhost {\r\n                               WindowsFeature IIS {\r\n                               Name = "Web-Server";\r\n            Ensure = "Present"\r\n        }\r\n    }\r\n}'}}})
    print(response)
if __name__ == '__main__':
    main()