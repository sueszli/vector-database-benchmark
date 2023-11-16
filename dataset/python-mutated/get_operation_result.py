from azure.identity import DefaultAzureCredential
from azure.mgmt.vmwarecloudsimple import VMwareCloudSimple
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-vmwarecloudsimple\n# USAGE\n    python get_operation_result.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = VMwareCloudSimple(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.operations.get(region_id='westus2', referer='https://management.azure.com/', operation_id='f8e1c8f1-7d52-11e9-8e07-9a86872085ff')
    print(response)
if __name__ == '__main__':
    main()