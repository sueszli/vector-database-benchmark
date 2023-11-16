from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python validate_app_service_certificate_purchase_information_by_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.app_service_certificate_orders.validate_purchase_information(app_service_certificate_order={'location': 'Global', 'properties': {'autoRenew': True, 'certificates': {'SampleCertName1': {'keyVaultId': '/subscriptions/34adfa4f-cedf-4dc0-ba29-b6d1a69ab345/resourcegroups/testrg123/providers/microsoft.keyvault/vaults/SamplevaultName', 'keyVaultSecretName': 'SampleSecretName1'}, 'SampleCertName2': {'keyVaultId': '/subscriptions/34adfa4f-cedf-4dc0-ba29-b6d1a69ab345/resourcegroups/testrg123/providers/microsoft.keyvault/vaults/SamplevaultName', 'keyVaultSecretName': 'SampleSecretName2'}}, 'distinguishedName': 'CN=SampleCustomDomain.com', 'keySize': 2048, 'productType': 'StandardDomainValidatedSsl', 'validityInYears': 2}})
    print(response)
if __name__ == '__main__':
    main()