from google.cloud import dataform_v1beta1

def sample_remove_directory():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.RemoveDirectoryRequest(workspace='workspace_value', path='path_value')
    client.remove_directory(request=request)