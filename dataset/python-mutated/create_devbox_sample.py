import logging
import os
from azure.developer.devcenter import DevCenterClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError
'\nFILE: create_devbox_sample.py\n\nDESCRIPTION:\n    This sample demonstrates how to create, connect and delete a dev box using python DevCenterClient. For this sample,\n    you must have previously configured DevCenter, Project, Network Connection, Dev Box Definition, and Pool.More details \n    on how to configure those requirements at https://learn.microsoft.com/azure/dev-box/quickstart-configure-dev-box-service\n\n\nUSAGE:\n    python create_devbox_sample.py\n\n    Set the environment variables with your own values before running the sample:\n    1) DEVCENTER_ENDPOINT - the endpoint for your devcenter\n'

def get_project_name(LOG, client):
    if False:
        print('Hello World!')
    projects = list(client.projects.list_by_dev_center(top=1))
    return projects[0].name

def main():
    if False:
        print('Hello World!')
    try:
        endpoint = os.environ['DEVCENTER_ENDPOINT']
    except KeyError:
        raise ValueError("Missing environment variable 'DEVCENTER_ENDPOINT' - please set it before running the example")
    client = DevCenterClient(endpoint, credential=DefaultAzureCredential())
    projects = list(client.list_projects(top=1))
    target_project_name = projects[0]['name']
    pools = list(client.list_pools(target_project_name, top=1))
    target_pool_name = pools[0]['name']
    create_response = client.begin_create_dev_box(target_project_name, 'me', 'Test_DevBox', {'poolName': target_pool_name})
    devbox_result = create_response.result()
    print(f"Provisioned dev box with status {devbox_result['provisioningState']}.")
    remote_connection_response = client.get_remote_connection(target_project_name, 'me', 'Test_DevBox')
    print(f"Connect to the dev box using web URL {remote_connection_response['webUrl']}")
    delete_response = client.begin_delete_dev_box(target_project_name, 'me', 'Test_DevBox')
    delete_response.wait()
    print('Deleted dev box successfully.')
if __name__ == '__main__':
    main()