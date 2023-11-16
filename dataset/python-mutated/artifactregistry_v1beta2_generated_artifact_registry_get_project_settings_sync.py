from google.cloud import artifactregistry_v1beta2

def sample_get_project_settings():
    if False:
        return 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.GetProjectSettingsRequest(name='name_value')
    response = client.get_project_settings(request=request)
    print(response)