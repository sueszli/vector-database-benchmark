from azure.identity import DefaultAzureCredential
from azure.mgmt.vmwarecloudsimple import VMwareCloudSimple
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-vmwarecloudsimple\n# USAGE\n    python get_virtual_machine_template.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = VMwareCloudSimple(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machine_templates.get(region_id='westus2', pc_name='myPrivateCloud', virtual_machine_template_name='vm-34')
    print(response)
if __name__ == '__main__':
    main()