from google.cloud import artifactregistry_v1

def sample_batch_delete_versions():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.BatchDeleteVersionsRequest(names=['names_value1', 'names_value2'])
    operation = client.batch_delete_versions(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)