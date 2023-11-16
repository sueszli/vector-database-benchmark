from google.cloud.devtools import cloudbuild_v2

def sample_get_repository():
    if False:
        print('Hello World!')
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.GetRepositoryRequest(name='name_value')
    response = client.get_repository(request=request)
    print(response)