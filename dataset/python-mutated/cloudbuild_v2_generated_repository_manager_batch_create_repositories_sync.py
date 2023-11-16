from google.cloud.devtools import cloudbuild_v2

def sample_batch_create_repositories():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v2.RepositoryManagerClient()
    requests = cloudbuild_v2.CreateRepositoryRequest()
    requests.parent = 'parent_value'
    requests.repository.remote_uri = 'remote_uri_value'
    requests.repository_id = 'repository_id_value'
    request = cloudbuild_v2.BatchCreateRepositoriesRequest(parent='parent_value', requests=requests)
    operation = client.batch_create_repositories(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)