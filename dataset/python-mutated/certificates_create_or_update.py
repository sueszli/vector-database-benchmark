from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python certificates_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.certificates.begin_create_or_update(resource_group_name='myResourceGroup', service_name='myservice', certificate_name='mycertificate', certificate_resource={'properties': {'certVersion': '08a219d06d874795a96db47e06fbb01e', 'keyVaultCertName': 'mycert', 'type': 'KeyVaultCertificate', 'vaultUri': 'https://myvault.vault.azure.net'}}).result()
    print(response)
if __name__ == '__main__':
    main()