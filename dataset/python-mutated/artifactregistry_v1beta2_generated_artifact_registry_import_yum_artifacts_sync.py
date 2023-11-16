from google.cloud import artifactregistry_v1beta2

def sample_import_yum_artifacts():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.ImportYumArtifactsRequest()
    operation = client.import_yum_artifacts(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)