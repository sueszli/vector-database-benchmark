from google.cloud import osconfig_v1

def sample_execute_patch_job():
    if False:
        print('Hello World!')
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.ExecutePatchJobRequest(parent='parent_value')
    response = client.execute_patch_job(request=request)
    print(response)