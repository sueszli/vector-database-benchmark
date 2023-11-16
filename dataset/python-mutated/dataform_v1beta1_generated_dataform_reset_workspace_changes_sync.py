from google.cloud import dataform_v1beta1

def sample_reset_workspace_changes():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ResetWorkspaceChangesRequest(name='name_value')
    client.reset_workspace_changes(request=request)