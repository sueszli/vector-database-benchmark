from google.cloud import dataform_v1beta1

def sample_delete_workspace():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.DeleteWorkspaceRequest(name='name_value')
    client.delete_workspace(request=request)