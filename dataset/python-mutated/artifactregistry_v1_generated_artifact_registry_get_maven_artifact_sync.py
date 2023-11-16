from google.cloud import artifactregistry_v1

def sample_get_maven_artifact():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetMavenArtifactRequest(name='name_value')
    response = client.get_maven_artifact(request=request)
    print(response)