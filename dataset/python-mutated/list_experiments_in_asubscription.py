from azure.identity import DefaultAzureCredential
from azure.mgmt.chaos import ChaosManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-chaos\n# USAGE\n    python list_experiments_in_asubscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ChaosManagementClient(credential=DefaultAzureCredential(), subscription_id='6b052e15-03d3-4f17-b2e1-be7f07588291')
    response = client.experiments.list_all()
    for item in response:
        print(item)
if __name__ == '__main__':
    main()