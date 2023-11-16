from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python validate_probe.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.validate_probe(validate_probe_input={'probeURL': 'https://www.bing.com/image'})
    print(response)
if __name__ == '__main__':
    main()