from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python managed_environment_diagnostics_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='f07f3711-b45e-40fe-a941-4e6d93f851e6')
    response = client.managed_environment_diagnostics.list_detectors(resource_group_name='mikono-workerapp-test-rg', environment_name='mikonokubeenv')
    print(response)
if __name__ == '__main__':
    main()