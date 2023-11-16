from google.cloud import osconfig_v1

def sample_pause_patch_deployment():
    if False:
        print('Hello World!')
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.PausePatchDeploymentRequest(name='name_value')
    response = client.pause_patch_deployment(request=request)
    print(response)