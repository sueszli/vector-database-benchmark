from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-authorization\n# USAGE\n    python role_assignments_create_for_resource.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AuthorizationManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.role_assignments.create(scope='subscriptions/a925f2f7-5c63-4b7b-8799-25a5f97bc3b2/resourceGroups/testrg/providers/Microsoft.DocumentDb/databaseAccounts/test-db-account', role_assignment_name='05c5a614-a7d6-4502-b150-c2fb455033ff', parameters={'properties': {'principalId': 'ce2ce14e-85d7-4629-bdbc-454d0519d987', 'principalType': 'User', 'roleDefinitionId': '/subscriptions/a925f2f7-5c63-4b7b-8799-25a5f97bc3b2/providers/Microsoft.Authorization/roleDefinitions/0b5fe924-9a61-425c-96af-cfe6e287ca2d'}})
    print(response)
if __name__ == '__main__':
    main()