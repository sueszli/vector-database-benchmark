from azure.identity import DefaultAzureCredential
from azure.mgmt.advisor import AdvisorManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-advisor\n# USAGE\n    python create_suppression.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AdvisorManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.suppressions.create(resource_uri='resourceUri', recommendation_id='recommendationId', name='suppressionName1', suppression_contract={'properties': {'ttl': '07:00:00:00'}})
    print(response)
if __name__ == '__main__':
    main()