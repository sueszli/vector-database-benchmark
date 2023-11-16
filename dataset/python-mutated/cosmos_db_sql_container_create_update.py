from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_sql_container_create_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.sql_resources.begin_create_update_sql_container(resource_group_name='rg1', account_name='ddb1', database_name='databaseName', container_name='containerName', create_update_sql_container_parameters={'location': 'West US', 'properties': {'options': {}, 'resource': {'clientEncryptionPolicy': {'includedPaths': [{'clientEncryptionKeyId': 'keyId', 'encryptionAlgorithm': 'AEAD_AES_256_CBC_HMAC_SHA256', 'encryptionType': 'Deterministic', 'path': '/path'}], 'policyFormatVersion': 2}, 'conflictResolutionPolicy': {'conflictResolutionPath': '/path', 'mode': 'LastWriterWins'}, 'defaultTtl': 100, 'id': 'containerName', 'indexingPolicy': {'automatic': True, 'excludedPaths': [], 'includedPaths': [{'indexes': [{'dataType': 'String', 'kind': 'Range', 'precision': -1}, {'dataType': 'Number', 'kind': 'Range', 'precision': -1}], 'path': '/*'}], 'indexingMode': 'consistent'}, 'partitionKey': {'kind': 'Hash', 'paths': ['/AccountNumber']}, 'uniqueKeyPolicy': {'uniqueKeys': [{'paths': ['/testPath']}]}}}, 'tags': {}}).result()
    print(response)
if __name__ == '__main__':
    main()