from google.cloud.devtools import cloudbuild_v2

def sample_create_connection():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.CreateConnectionRequest(parent='parent_value', connection_id='connection_id_value')
    operation = client.create_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)