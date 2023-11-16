from google.cloud.devtools import cloudbuild_v2

def sample_fetch_read_token():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.FetchReadTokenRequest(repository='repository_value')
    response = client.fetch_read_token(request=request)
    print(response)