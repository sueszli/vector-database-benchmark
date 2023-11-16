from google.cloud import osconfig_v1

def sample_delete_patch_deployment():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.DeletePatchDeploymentRequest(name='name_value')
    client.delete_patch_deployment(request=request)