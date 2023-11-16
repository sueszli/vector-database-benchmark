from azure.identity import DefaultAzureCredential
from azure.mgmt.batch import BatchManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-batch\n# USAGE\n    python pool_create_virtual_machine_configuration_extensions.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = BatchManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.pool.create(resource_group_name='default-azurebatch-japaneast', account_name='sampleacct', pool_name='testpool', parameters={'properties': {'deploymentConfiguration': {'virtualMachineConfiguration': {'extensions': [{'autoUpgradeMinorVersion': True, 'enableAutomaticUpgrade': True, 'name': 'batchextension1', 'publisher': 'Microsoft.Azure.KeyVault', 'settings': {'authenticationSettingsKey': 'authenticationSettingsValue', 'secretsManagementSettingsKey': 'secretsManagementSettingsValue'}, 'type': 'KeyVaultForLinux', 'typeHandlerVersion': '2.0'}], 'imageReference': {'offer': '0001-com-ubuntu-server-focal', 'publisher': 'Canonical', 'sku': '20_04-lts'}, 'nodeAgentSkuId': 'batch.node.ubuntu 20.04'}}, 'scaleSettings': {'autoScale': {'evaluationInterval': 'PT5M', 'formula': '$TargetDedicatedNodes=1'}}, 'targetNodeCommunicationMode': 'Default', 'vmSize': 'STANDARD_D4'}})
    print(response)
if __name__ == '__main__':
    main()