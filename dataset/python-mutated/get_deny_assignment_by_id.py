from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-authorization\n# USAGE\n    python get_deny_assignment_by_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AuthorizationManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.deny_assignments.get_by_id(deny_assignment_id='subscriptions/subId/resourcegroups/rgname/providers/Microsoft.Authorization/denyAssignments/daId')
    print(response)
if __name__ == '__main__':
    main()