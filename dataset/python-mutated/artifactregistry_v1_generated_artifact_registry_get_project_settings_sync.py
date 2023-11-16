from google.cloud import artifactregistry_v1

def sample_get_project_settings():
    if False:
        return 10
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetProjectSettingsRequest(name='name_value')
    response = client.get_project_settings(request=request)
    print(response)