from google.cloud.devtools import cloudbuild_v2

def sample_fetch_read_write_token():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.FetchReadWriteTokenRequest(repository='repository_value')
    response = client.fetch_read_write_token(request=request)
    print(response)