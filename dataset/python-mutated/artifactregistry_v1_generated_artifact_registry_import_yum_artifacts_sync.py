from google.cloud import artifactregistry_v1

def sample_import_yum_artifacts():
    if False:
        return 10
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ImportYumArtifactsRequest()
    operation = client.import_yum_artifacts(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)