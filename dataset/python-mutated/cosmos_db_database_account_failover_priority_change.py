from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_database_account_failover_priority_change.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    client.database_accounts.begin_failover_priority_change(resource_group_name='rg1', account_name='ddb1-failover', failover_parameters={'failoverPolicies': [{'failoverPriority': 0, 'locationName': 'eastus'}, {'failoverPriority': 1, 'locationName': 'westus'}]}).result()
if __name__ == '__main__':
    main()