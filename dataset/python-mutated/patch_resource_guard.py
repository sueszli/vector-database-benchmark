from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python patch_resource_guard.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='0b352192-dcac-4cc7-992e-a96190ccc68c')
    response = client.resource_guards.patch(resource_group_name='SampleResourceGroup', resource_guards_name='swaggerExample', parameters={'tags': {'newKey': 'newVal'}})
    print(response)
if __name__ == '__main__':
    main()