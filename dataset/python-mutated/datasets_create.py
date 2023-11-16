from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datafactory\n# USAGE\n    python datasets_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DataFactoryManagementClient(credential=DefaultAzureCredential(), subscription_id='12345678-1234-1234-1234-12345678abc')
    response = client.datasets.create_or_update(resource_group_name='exampleResourceGroup', factory_name='exampleFactoryName', dataset_name='exampleDataset', dataset={'properties': {'linkedServiceName': {'referenceName': 'exampleLinkedService', 'type': 'LinkedServiceReference'}, 'parameters': {'MyFileName': {'type': 'String'}, 'MyFolderPath': {'type': 'String'}}, 'type': 'AzureBlob', 'typeProperties': {'fileName': {'type': 'Expression', 'value': '@dataset().MyFileName'}, 'folderPath': {'type': 'Expression', 'value': '@dataset().MyFolderPath'}, 'format': {'type': 'TextFormat'}}}})
    print(response)
if __name__ == '__main__':
    main()