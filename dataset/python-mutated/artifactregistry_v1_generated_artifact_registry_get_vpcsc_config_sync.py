from google.cloud import artifactregistry_v1

def sample_get_vpcsc_config():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetVPCSCConfigRequest(name='name_value')
    response = client.get_vpcsc_config(request=request)
    print(response)