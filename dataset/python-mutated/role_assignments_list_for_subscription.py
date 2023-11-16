from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-authorization\n# USAGE\n    python role_assignments_list_for_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AuthorizationManagementClient(credential=DefaultAzureCredential(), subscription_id='a925f2f7-5c63-4b7b-8799-25a5f97bc3b2')
    response = client.role_assignments.list_for_subscription()
    for item in response:
        print(item)
if __name__ == '__main__':
    main()