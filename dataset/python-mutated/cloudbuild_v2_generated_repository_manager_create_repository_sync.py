from google.cloud.devtools import cloudbuild_v2

def sample_create_repository():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v2.RepositoryManagerClient()
    repository = cloudbuild_v2.Repository()
    repository.remote_uri = 'remote_uri_value'
    request = cloudbuild_v2.CreateRepositoryRequest(parent='parent_value', repository=repository, repository_id='repository_id_value')
    operation = client.create_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)