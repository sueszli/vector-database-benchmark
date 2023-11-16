from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import AuthenticationException
from azureml.core import Workspace

def get_auth():
    if False:
        while True:
            i = 10
    '\n    Method to get the correct Azure ML Authentication type\n\n    Always start with CLI Authentication and if it fails, fall back\n    to interactive login\n    '
    try:
        auth_type = AzureCliAuthentication()
        auth_type.get_authentication_header()
    except AuthenticationException:
        auth_type = InteractiveLoginAuthentication()
    return auth_type

def get_or_create_workspace(subscription_id: str, resource_group: str, workspace_name: str, workspace_region: str) -> Workspace:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns workspace if one exists already with the name\n    otherwise creates a new one.\n\n    Args\n    subscription_id: Azure subscription id\n    resource_group: Azure resource group to create workspace and related resources\n    workspace_name: name of azure ml workspac\n    workspace_region: region for workspace\n    '
    try:
        ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, auth=get_auth())
    except Exception:
        print('Creating new workspace')
        ws = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, create_resource_group=True, location=workspace_region, auth=get_auth())
    return ws