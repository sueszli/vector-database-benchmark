from google.cloud.devtools import cloudbuild_v2

def sample_get_connection():
    if False:
        return 10
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.GetConnectionRequest(name='name_value')
    response = client.get_connection(request=request)
    print(response)