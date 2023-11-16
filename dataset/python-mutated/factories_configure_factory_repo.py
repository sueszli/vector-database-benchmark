from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datafactory\n# USAGE\n    python factories_configure_factory_repo.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataFactoryManagementClient(credential=DefaultAzureCredential(), subscription_id='12345678-1234-1234-1234-12345678abc')
    response = client.factories.configure_factory_repo(location_id='East US', factory_repo_update={'factoryResourceId': '/subscriptions/12345678-1234-1234-1234-12345678abc/resourceGroups/exampleResourceGroup/providers/Microsoft.DataFactory/factories/exampleFactoryName', 'repoConfiguration': {'accountName': 'ADF', 'collaborationBranch': 'master', 'lastCommitId': '', 'projectName': 'project', 'repositoryName': 'repo', 'rootFolder': '/', 'tenantId': '', 'type': 'FactoryVSTSConfiguration'}})
    print(response)
if __name__ == '__main__':
    main()