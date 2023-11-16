from azure.identity import DefaultAzureCredential
from azure.mgmt.avs import AVSClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-avs\n# USAGE\n    python script_executions_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AVSClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.script_executions.begin_create_or_update(resource_group_name='group1', private_cloud_name='cloud1', script_execution_name='addSsoServer', script_execution={'properties': {'hiddenParameters': [{'name': 'Password', 'secureValue': 'PlaceholderPassword', 'type': 'SecureValue'}], 'parameters': [{'name': 'DomainName', 'type': 'Value', 'value': 'placeholderDomain.local'}, {'name': 'BaseUserDN', 'type': 'Value', 'value': 'DC=placeholder, DC=placeholder'}], 'retention': 'P0Y0M60DT0H60M60S', 'scriptCmdletId': '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/group1/providers/Microsoft.AVS/privateClouds/cloud1/scriptPackages/AVS.PowerCommands@1.0.0/scriptCmdlets/New-SsoExternalIdentitySource', 'timeout': 'P0Y0M0DT0H60M60S'}}).result()
    print(response)
if __name__ == '__main__':
    main()