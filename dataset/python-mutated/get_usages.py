from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cognitiveservices\n# USAGE\n    python get_usages.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CognitiveServicesManagementClient(credential=DefaultAzureCredential(), subscription_id='5a4f5c2e-6983-4ccb-bd34-2196d5b5bbd3')
    response = client.accounts.list_usages(resource_group_name='myResourceGroup', account_name='TestUsage02')
    print(response)
if __name__ == '__main__':
    main()