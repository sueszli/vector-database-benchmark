from azure.identity import DefaultAzureCredential
from azure.mgmt.vmwarecloudsimple import VMwareCloudSimple
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-vmwarecloudsimple\n# USAGE\n    python list_rg_virtual_machines.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = VMwareCloudSimple(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.list_by_resource_group(resource_group_name='myResourceGroup')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()