from azure.identity import DefaultAzureCredential
from azure.mgmt.chaos import ChaosManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-chaos\n# USAGE\n    python get_aexperiment_execution_details.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ChaosManagementClient(credential=DefaultAzureCredential(), subscription_id='6b052e15-03d3-4f17-b2e1-be7f07588291')
    response = client.experiments.get_execution_details(resource_group_name='exampleRG', experiment_name='exampleExperiment', execution_details_id='f24500ad-744e-4a26-864b-b76199eac333')
    print(response)
if __name__ == '__main__':
    main()