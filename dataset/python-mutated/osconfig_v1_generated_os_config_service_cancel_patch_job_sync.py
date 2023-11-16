from google.cloud import osconfig_v1

def sample_cancel_patch_job():
    if False:
        i = 10
        return i + 15
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.CancelPatchJobRequest(name='name_value')
    response = client.cancel_patch_job(request=request)
    print(response)