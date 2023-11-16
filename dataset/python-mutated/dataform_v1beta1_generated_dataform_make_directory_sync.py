from google.cloud import dataform_v1beta1

def sample_make_directory():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.MakeDirectoryRequest(workspace='workspace_value', path='path_value')
    response = client.make_directory(request=request)
    print(response)