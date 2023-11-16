from azure.identity import DefaultAzureCredential
from azure.mgmt.customproviders import Customproviders
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-customproviders\n# USAGE\n    python get_a_custom_resource_provider.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = Customproviders(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.custom_resource_provider.get(resource_group_name='testRG', resource_provider_name='newrp')
    print(response)
if __name__ == '__main__':
    main()