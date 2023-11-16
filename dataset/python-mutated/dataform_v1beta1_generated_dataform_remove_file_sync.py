from google.cloud import dataform_v1beta1

def sample_remove_file():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.RemoveFileRequest(workspace='workspace_value', path='path_value')
    client.remove_file(request=request)