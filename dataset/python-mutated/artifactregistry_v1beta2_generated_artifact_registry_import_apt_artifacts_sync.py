from google.cloud import artifactregistry_v1beta2

def sample_import_apt_artifacts():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.ImportAptArtifactsRequest()
    operation = client.import_apt_artifacts(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)