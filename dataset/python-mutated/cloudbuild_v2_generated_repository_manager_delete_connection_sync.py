from google.cloud.devtools import cloudbuild_v2

def sample_delete_connection():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.DeleteConnectionRequest(name='name_value')
    operation = client.delete_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)