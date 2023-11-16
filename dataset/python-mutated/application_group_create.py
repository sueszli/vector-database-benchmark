from azure.identity import DefaultAzureCredential
from azure.mgmt.desktopvirtualization import DesktopVirtualizationMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-desktopvirtualization\n# USAGE\n    python application_group_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DesktopVirtualizationMgmtClient(credential=DefaultAzureCredential(), subscription_id='daefabc0-95b4-48b3-b645-8a753a63c4fa')
    response = client.application_groups.create_or_update(resource_group_name='resourceGroup1', application_group_name='applicationGroup1', application_group={'location': 'centralus', 'properties': {'applicationGroupType': 'RemoteApp', 'description': 'des1', 'friendlyName': 'friendly', 'hostPoolArmPath': '/subscriptions/daefabc0-95b4-48b3-b645-8a753a63c4fa/resourceGroups/resourceGroup1/providers/Microsoft.DesktopVirtualization/hostPools/hostPool1', 'showInFeed': True}, 'tags': {'tag1': 'value1', 'tag2': 'value2'}})
    print(response)
if __name__ == '__main__':
    main()