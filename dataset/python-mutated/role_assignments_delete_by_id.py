from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-authorization\n# USAGE\n    python role_assignments_delete_by_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AuthorizationManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.role_assignments.delete_by_id(role_assignment_id='subscriptions/a925f2f7-5c63-4b7b-8799-25a5f97bc3b2/providers/Microsoft.Authorization/roleAssignments/b0f43c54-e787-4862-89b1-a653fa9cf747')
    print(response)
if __name__ == '__main__':
    main()