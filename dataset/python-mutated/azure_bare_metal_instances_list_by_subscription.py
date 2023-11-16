from azure.identity import DefaultAzureCredential
from azure.mgmt.baremetalinfrastructure import BareMetalInfrastructureClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-baremetalinfrastructure\n# USAGE\n    python azure_bare_metal_instances_list_by_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = BareMetalInfrastructureClient(credential=DefaultAzureCredential(), subscription_id='f0f4887f-d13c-4943-a8ba-d7da28d2a3fd')
    response = client.azure_bare_metal_instances.list_by_subscription()
    for item in response:
        print(item)
if __name__ == '__main__':
    main()