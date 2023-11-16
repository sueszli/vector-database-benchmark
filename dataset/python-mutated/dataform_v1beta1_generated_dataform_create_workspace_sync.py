from google.cloud import dataform_v1beta1

def sample_create_workspace():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.CreateWorkspaceRequest(parent='parent_value', workspace_id='workspace_id_value')
    response = client.create_workspace(request=request)
    print(response)