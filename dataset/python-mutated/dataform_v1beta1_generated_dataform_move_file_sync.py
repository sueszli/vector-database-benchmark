from google.cloud import dataform_v1beta1

def sample_move_file():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.MoveFileRequest(workspace='workspace_value', path='path_value', new_path='new_path_value')
    response = client.move_file(request=request)
    print(response)