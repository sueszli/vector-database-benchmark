from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python afd_custom_domains_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.afd_custom_domains.begin_update(resource_group_name='RG', profile_name='profile1', custom_domain_name='domain1', custom_domain_update_properties={'properties': {'azureDnsZone': {'id': ''}, 'tlsSettings': {'certificateType': 'CustomerCertificate', 'minimumTlsVersion': 'TLS12'}}}).result()
    print(response)
if __name__ == '__main__':
    main()