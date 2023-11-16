from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservicefleet import ContainerServiceFleetMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservicefleet\n# USAGE\n    python fleets_list_credentials_result.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ContainerServiceFleetMgmtClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.fleets.list_credentials(resource_group_name='rg1', fleet_name='fleet')
    print(response)
if __name__ == '__main__':
    main()