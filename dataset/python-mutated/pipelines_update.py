from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datafactory\n# USAGE\n    python pipelines_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataFactoryManagementClient(credential=DefaultAzureCredential(), subscription_id='12345678-1234-1234-1234-12345678abc')
    response = client.pipelines.create_or_update(resource_group_name='exampleResourceGroup', factory_name='exampleFactoryName', pipeline_name='examplePipeline', pipeline={'properties': {'activities': [{'name': 'ExampleForeachActivity', 'type': 'ForEach', 'typeProperties': {'activities': [{'inputs': [{'parameters': {'MyFileName': 'examplecontainer.csv', 'MyFolderPath': 'examplecontainer'}, 'referenceName': 'exampleDataset', 'type': 'DatasetReference'}], 'name': 'ExampleCopyActivity', 'outputs': [{'parameters': {'MyFileName': {'type': 'Expression', 'value': '@item()'}, 'MyFolderPath': 'examplecontainer'}, 'referenceName': 'exampleDataset', 'type': 'DatasetReference'}], 'type': 'Copy', 'typeProperties': {'dataIntegrationUnits': 32, 'sink': {'type': 'BlobSink'}, 'source': {'type': 'BlobSource'}}}], 'isSequential': True, 'items': {'type': 'Expression', 'value': '@pipeline().parameters.OutputBlobNameList'}}}], 'description': 'Example description', 'parameters': {'OutputBlobNameList': {'type': 'Array'}}, 'policy': {'elapsedTimeMetric': {'duration': '0.00:10:00'}}}})
    print(response)
if __name__ == '__main__':
    main()