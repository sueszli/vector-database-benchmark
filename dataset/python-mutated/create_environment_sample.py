import logging
import os
from azure.developer.devcenter import DevCenterClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError
'\nFILE: create_environment_sample.py\n\nDESCRIPTION:\n    This sample demonstrates how to create and delete Environments using python DevCenterClient. For this sample,\n    you must have previously configured a DevCenter, Project, Catalog, and Environment Type. More details \n    on how to configure those requirements at https://learn.microsoft.com/azure/deployment-environments/\n\nUSAGE:\n    python create_environment_sample.py\n\n    Set the environment variables with your own values before running the sample:\n    1) DEVCENTER_ENDPOINT - the endpoint for your devcenter\n'

def main():
    if False:
        while True:
            i = 10
    try:
        endpoint = os.environ['DEVCENTER_ENDPOINT']
    except KeyError:
        raise ValueError("Missing environment variable 'DEVCENTER_ENDPOINT' - please set it before running the example")
    client = DevCenterClient(endpoint, credential=DefaultAzureCredential())
    target_project_name = list(client.list_projects(top=1))[0]['name']
    target_catalog_name = list(client.list_catalogs(target_project_name, top=1))[0]['name']
    target_environment_definition_name = list(client.list_environment_definitions_by_catalog(target_project_name, target_catalog_name, top=1))[0]['name']
    target_environment_type_name = list(client.list_environment_types(target_project_name, top=1))[0]['name']
    environment = {'catalogName': target_catalog_name, 'environmentDefinitionName': target_environment_definition_name, 'environmentType': target_environment_type_name}
    create_response = client.begin_create_or_update_environment(target_project_name, 'me', 'DevTestEnv', environment)
    environment_result = create_response.result()
    print(f"Provisioned environment with status {environment_result['provisioningState']}.")
    delete_response = client.begin_delete_environment(target_project_name, 'me', 'DevTestEnv')
    delete_result = delete_response.result()
    print(f"Completed deletion for the environment with status {delete_result['status']}")
if __name__ == '__main__':
    main()