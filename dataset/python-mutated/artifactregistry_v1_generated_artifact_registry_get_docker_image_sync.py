from google.cloud import artifactregistry_v1

def sample_get_docker_image():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetDockerImageRequest(name='name_value')
    response = client.get_docker_image(request=request)
    print(response)