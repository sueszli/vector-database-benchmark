from google.cloud import dataform_v1beta1

def sample_read_file():
    if False:
        print('Hello World!')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ReadFileRequest(workspace='workspace_value', path='path_value')
    response = client.read_file(request=request)
    print(response)