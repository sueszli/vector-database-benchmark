from azure.identity import DefaultAzureCredential
from azure.mgmt.databoxedge import DataBoxEdgeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databoxedge\n# USAGE\n    python share_put.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataBoxEdgeManagementClient(credential=DefaultAzureCredential(), subscription_id='4385cf00-2d3a-425a-832f-f4285b1c9dce')
    response = client.shares.begin_create_or_update(device_name='testedgedevice', name='smbshare', resource_group_name='GroupForEdgeAutomation', share={'properties': {'accessProtocol': 'SMB', 'azureContainerInfo': {'containerName': 'testContainerSMB', 'dataFormat': 'BlockBlob', 'storageAccountCredentialId': '/subscriptions/4385cf00-2d3a-425a-832f-f4285b1c9dce/resourceGroups/GroupForEdgeAutomation/providers/Microsoft.DataBoxEdge/dataBoxEdgeDevices/testedgedevice/storageAccountCredentials/sac1'}, 'dataPolicy': 'Cloud', 'description': '', 'monitoringStatus': 'Enabled', 'shareStatus': 'Online', 'userAccessRights': [{'accessType': 'Change', 'userId': '/subscriptions/4385cf00-2d3a-425a-832f-f4285b1c9dce/resourceGroups/GroupForEdgeAutomation/providers/Microsoft.DataBoxEdge/dataBoxEdgeDevices/testedgedevice/users/user2'}]}}).result()
    print(response)
if __name__ == '__main__':
    main()