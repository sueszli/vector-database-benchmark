from google.cloud import osconfig_v1

def sample_get_patch_job():
    if False:
        while True:
            i = 10
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.GetPatchJobRequest(name='name_value')
    response = client.get_patch_job(request=request)
    print(response)