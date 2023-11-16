from google.cloud import artifactregistry_v1beta2

def sample_update_project_settings():
    if False:
        print('Hello World!')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.UpdateProjectSettingsRequest()
    response = client.update_project_settings(request=request)
    print(response)