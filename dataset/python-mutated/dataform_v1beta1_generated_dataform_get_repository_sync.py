from google.cloud import dataform_v1beta1

def sample_get_repository():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.GetRepositoryRequest(name='name_value')
    response = client.get_repository(request=request)
    print(response)