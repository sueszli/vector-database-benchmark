from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_backend_service_fabric.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.backend.create_or_update(resource_group_name='rg1', service_name='apimService1', backend_id='sfbackend', parameters={'properties': {'description': 'Service Fabric Test App 1', 'properties': {'serviceFabricCluster': {'clientCertificateId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.ApiManagement/service/apimService1/certificates/cert1', 'managementEndpoints': ['https://somecluster.com'], 'maxPartitionResolutionRetries': 5, 'serverX509Names': [{'issuerCertificateThumbprint': 'IssuerCertificateThumbprint1', 'name': 'ServerCommonName1'}]}}, 'protocol': 'http', 'url': 'fabric:/mytestapp/mytestservice'}})
    print(response)
if __name__ == '__main__':
    main()