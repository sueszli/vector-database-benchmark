from google.cloud import osconfig_v1

def sample_update_patch_deployment():
    if False:
        i = 10
        return i + 15
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.UpdatePatchDeploymentRequest()
    response = client.update_patch_deployment(request=request)
    print(response)