from azure.identity import DefaultAzureCredential
from azure.mgmt.deploymentmanager import AzureDeploymentManager
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-deploymentmanager\n# USAGE\n    python list_steps.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AzureDeploymentManager(credential=DefaultAzureCredential(), subscription_id='caac1590-e859-444f-a9e0-62091c0f5929')
    response = client.steps.list(resource_group_name='myResourceGroup')
    print(response)
if __name__ == '__main__':
    main()