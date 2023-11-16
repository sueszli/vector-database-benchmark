from google.cloud import artifactregistry_v1

def sample_import_apt_artifacts():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ImportAptArtifactsRequest()
    operation = client.import_apt_artifacts(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)