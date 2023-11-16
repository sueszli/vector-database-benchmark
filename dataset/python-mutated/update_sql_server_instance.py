from azure.identity import DefaultAzureCredential
from azure.mgmt.azurearcdata import AzureArcDataManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-azurearcdata\n# USAGE\n    python update_sql_server_instance.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AzureArcDataManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-1111-2222-3333-444444444444')
    response = client.sql_server_instances.update(resource_group_name='testrg', sql_server_instance_name='testsqlServerInstance', parameters={'tags': {'mytag': 'myval'}})
    print(response)
if __name__ == '__main__':
    main()