from google.cloud import artifactregistry_v1

def sample_delete_tag():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.DeleteTagRequest()
    client.delete_tag(request=request)