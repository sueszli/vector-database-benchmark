from google.cloud import osconfig_v1

def sample_create_patch_deployment():
    if False:
        for i in range(10):
            print('nop')
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.CreatePatchDeploymentRequest(parent='parent_value', patch_deployment_id='patch_deployment_id_value')
    response = client.create_patch_deployment(request=request)
    print(response)