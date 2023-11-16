from azure.identity import DefaultAzureCredential
from azure.mgmt.chaos import ChaosManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-chaos\n# USAGE\n    python get_aexperiment_status.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ChaosManagementClient(credential=DefaultAzureCredential(), subscription_id='6b052e15-03d3-4f17-b2e1-be7f07588291')
    response = client.experiments.get_status(resource_group_name='exampleRG', experiment_name='exampleExperiment', status_id='50734542-2e64-4e08-814c-cc0e7475f7e4')
    print(response)
if __name__ == '__main__':
    main()