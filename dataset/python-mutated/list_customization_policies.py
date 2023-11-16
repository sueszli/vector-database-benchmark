from azure.identity import DefaultAzureCredential
from azure.mgmt.vmwarecloudsimple import VMwareCloudSimple
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-vmwarecloudsimple\n# USAGE\n    python list_customization_policies.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = VMwareCloudSimple(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.customization_policies.list(region_id='myResourceGroup', pc_name='myPrivateCloud')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()