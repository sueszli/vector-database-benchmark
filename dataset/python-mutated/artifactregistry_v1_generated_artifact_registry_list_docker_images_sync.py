from google.cloud import artifactregistry_v1

def sample_list_docker_images():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ListDockerImagesRequest(parent='parent_value')
    page_result = client.list_docker_images(request=request)
    for response in page_result:
        print(response)