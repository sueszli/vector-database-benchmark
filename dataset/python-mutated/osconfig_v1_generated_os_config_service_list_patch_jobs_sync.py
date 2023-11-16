from google.cloud import osconfig_v1

def sample_list_patch_jobs():
    if False:
        return 10
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.ListPatchJobsRequest(parent='parent_value')
    page_result = client.list_patch_jobs(request=request)
    for response in page_result:
        print(response)