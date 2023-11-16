from google.cloud import artifactregistry_v1

def sample_update_vpcsc_config():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.UpdateVPCSCConfigRequest()
    response = client.update_vpcsc_config(request=request)
    print(response)