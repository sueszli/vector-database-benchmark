from azure.identity import DefaultAzureCredential
from azure.mgmt.confluent import ConfluentManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-confluent\n# USAGE\n    python validations_validate_organizations.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ConfluentManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.validations.validate_organization(resource_group_name='myResourceGroup', organization_name='myOrganization', body={'location': 'West US', 'properties': {'offerDetail': {'id': 'string', 'planId': 'string', 'planName': 'string', 'publisherId': 'string', 'termUnit': 'string'}, 'userDetail': {'emailAddress': 'abc@microsoft.com', 'firstName': 'string', 'lastName': 'string'}}, 'tags': {'Environment': 'Dev'}})
    print(response)
if __name__ == '__main__':
    main()