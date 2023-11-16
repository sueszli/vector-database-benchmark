from google.cloud import osconfig_v1

def sample_list_patch_job_instance_details():
    if False:
        print('Hello World!')
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.ListPatchJobInstanceDetailsRequest(parent='parent_value')
    page_result = client.list_patch_job_instance_details(request=request)
    for response in page_result:
        print(response)