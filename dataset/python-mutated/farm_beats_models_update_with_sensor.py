from azure.identity import DefaultAzureCredential
from azure.mgmt.agrifood import AgriFoodMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-agrifood\n# USAGE\n    python farm_beats_models_update_with_sensor.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AgriFoodMgmtClient(credential=DefaultAzureCredential(), solution_id='SOLUTION_ID', subscription_id='11111111-2222-3333-4444-555555555555')
    response = client.farm_beats_models.begin_update(resource_group_name='examples-rg', farm_beats_resource_name='examples-farmBeatsResourceName', body={'identity': {'type': 'SystemAssigned'}, 'properties': {'sensorIntegration': {'enabled': 'True'}}, 'tags': {'key1': 'value1', 'key2': 'value2'}}).result()
    print(response)
if __name__ == '__main__':
    main()