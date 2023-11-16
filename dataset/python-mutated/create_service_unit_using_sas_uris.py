from azure.identity import DefaultAzureCredential
from azure.mgmt.deploymentmanager import AzureDeploymentManager
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-deploymentmanager\n# USAGE\n    python create_service_unit_using_sas_uris.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AzureDeploymentManager(credential=DefaultAzureCredential(), subscription_id='caac1590-e859-444f-a9e0-62091c0f5929')
    response = client.service_units.begin_create_or_update(resource_group_name='myResourceGroup', service_topology_name='myTopology', service_name='myService', service_unit_name='myServiceUnit', service_unit_info={'location': 'centralus', 'properties': {'artifacts': {'parametersUri': 'https://mystorageaccount.blob.core.windows.net/myartifactsource/parameter/myTopologyUnit.parameters.json?st=2018-07-07T14%3A10%3A00Z&se=2019-12-31T15%3A10%3A00Z&sp=rl&sv=2017-04-17&sr=c&sig=Yh2SoJ1NhhLRwCLln7de%2Fkabcdefghijklmno5sWEIk%3D', 'templateUri': 'https://mystorageaccount.blob.core.windows.net/myartifactsource/templates/myTopologyUnit.template.json?st=2018-07-07T14%3A10%3A00Z&se=2019-12-31T15%3A10%3A00Z&sp=rl&sv=2017-04-17&sr=c&sig=Yh2SoJ1NhhLRwCLln7de%2Fkabcdefghijklmno5sWEIk%3D'}, 'deploymentMode': 'Incremental', 'targetResourceGroup': 'myDeploymentResourceGroup'}, 'tags': {}}).result()
    print(response)
if __name__ == '__main__':
    main()