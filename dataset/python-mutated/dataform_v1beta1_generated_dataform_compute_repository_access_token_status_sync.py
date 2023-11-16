from google.cloud import dataform_v1beta1

def sample_compute_repository_access_token_status():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ComputeRepositoryAccessTokenStatusRequest(name='name_value')
    response = client.compute_repository_access_token_status(request=request)
    print(response)