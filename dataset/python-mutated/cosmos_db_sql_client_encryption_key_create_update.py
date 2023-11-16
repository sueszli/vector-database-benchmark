from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_sql_client_encryption_key_create_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subId')
    response = client.sql_resources.begin_create_update_client_encryption_key(resource_group_name='rgName', account_name='accountName', database_name='databaseName', client_encryption_key_name='cekName', create_update_client_encryption_key_parameters={'properties': {'resource': {'encryptionAlgorithm': 'AEAD_AES_256_CBC_HMAC_SHA256', 'id': 'cekName', 'keyWrapMetadata': {'algorithm': 'RSA-OAEP', 'name': 'customerManagedKey', 'type': 'AzureKeyVault', 'value': 'AzureKeyVault Key URL'}, 'wrappedDataEncryptionKey': 'U3dhZ2dlciByb2Nrcw=='}}}).result()
    print(response)
if __name__ == '__main__':
    main()