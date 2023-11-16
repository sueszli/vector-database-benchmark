from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_sql_role_assignment_create_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='mySubscriptionId')
    response = client.sql_resources.begin_create_update_sql_role_assignment(role_assignment_id='myRoleAssignmentId', resource_group_name='myResourceGroupName', account_name='myAccountName', create_update_sql_role_assignment_parameters={'properties': {'principalId': 'myPrincipalId', 'roleDefinitionId': '/subscriptions/mySubscriptionId/resourceGroups/myResourceGroupName/providers/Microsoft.DocumentDB/databaseAccounts/myAccountName/sqlRoleDefinitions/myRoleDefinitionId', 'scope': '/subscriptions/mySubscriptionId/resourceGroups/myResourceGroupName/providers/Microsoft.DocumentDB/databaseAccounts/myAccountName/dbs/purchases/colls/redmond-purchases'}}).result()
    print(response)
if __name__ == '__main__':
    main()