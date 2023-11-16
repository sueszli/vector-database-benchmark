from google.cloud import dataform_v1beta1

def sample_move_directory():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.MoveDirectoryRequest(workspace='workspace_value', path='path_value', new_path='new_path_value')
    response = client.move_directory(request=request)
    print(response)