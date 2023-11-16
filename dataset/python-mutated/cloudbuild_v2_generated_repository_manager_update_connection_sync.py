from google.cloud.devtools import cloudbuild_v2

def sample_update_connection():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.UpdateConnectionRequest()
    operation = client.update_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)