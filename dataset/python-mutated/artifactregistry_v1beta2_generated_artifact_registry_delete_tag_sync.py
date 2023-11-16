from google.cloud import artifactregistry_v1beta2

def sample_delete_tag():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.DeleteTagRequest()
    client.delete_tag(request=request)