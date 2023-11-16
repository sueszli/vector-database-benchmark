from google.cloud import dataform_v1beta1

def sample_write_file():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.WriteFileRequest(workspace='workspace_value', path='path_value', contents=b'contents_blob')
    response = client.write_file(request=request)
    print(response)