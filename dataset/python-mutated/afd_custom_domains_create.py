from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python afd_custom_domains_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.afd_custom_domains.begin_create(resource_group_name='RG', profile_name='profile1', custom_domain_name='domain1', custom_domain={'properties': {'azureDnsZone': {'id': ''}, 'hostName': 'www.someDomain.net', 'tlsSettings': {'certificateType': 'ManagedCertificate', 'minimumTlsVersion': 'TLS12'}}}).result()
    print(response)
if __name__ == '__main__':
    main()