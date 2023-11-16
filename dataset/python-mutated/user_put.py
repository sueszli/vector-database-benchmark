from azure.identity import DefaultAzureCredential
from azure.mgmt.databoxedge import DataBoxEdgeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databoxedge\n# USAGE\n    python user_put.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = DataBoxEdgeManagementClient(credential=DefaultAzureCredential(), subscription_id='4385cf00-2d3a-425a-832f-f4285b1c9dce')
    response = client.users.begin_create_or_update(device_name='testedgedevice', name='user1', resource_group_name='GroupForEdgeAutomation', user={'properties': {'encryptedPassword': {'encryptionAlgorithm': 'None', 'encryptionCertThumbprint': 'blah', 'value': '<value>'}, 'userType': 'Share'}}).result()
    print(response)
if __name__ == '__main__':
    main()