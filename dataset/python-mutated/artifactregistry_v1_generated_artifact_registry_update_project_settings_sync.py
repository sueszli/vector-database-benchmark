from google.cloud import artifactregistry_v1

def sample_update_project_settings():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.UpdateProjectSettingsRequest()
    response = client.update_project_settings(request=request)
    print(response)