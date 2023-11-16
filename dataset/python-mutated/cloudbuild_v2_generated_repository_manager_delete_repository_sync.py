from google.cloud.devtools import cloudbuild_v2

def sample_delete_repository():
    if False:
        while True:
            i = 10
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.DeleteRepositoryRequest(name='name_value')
    operation = client.delete_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)