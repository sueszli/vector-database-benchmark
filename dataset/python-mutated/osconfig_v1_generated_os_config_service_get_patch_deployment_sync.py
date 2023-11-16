from google.cloud import osconfig_v1

def sample_get_patch_deployment():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.GetPatchDeploymentRequest(name='name_value')
    response = client.get_patch_deployment(request=request)
    print(response)