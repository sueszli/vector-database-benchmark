from google.cloud import dataform_v1beta1

def sample_read_repository_file():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ReadRepositoryFileRequest(name='name_value', path='path_value')
    response = client.read_repository_file(request=request)
    print(response)