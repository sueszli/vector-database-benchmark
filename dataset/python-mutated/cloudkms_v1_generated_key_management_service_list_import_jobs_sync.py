from google.cloud import kms_v1

def sample_list_import_jobs():
    if False:
        i = 10
        return i + 15
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.ListImportJobsRequest(parent='parent_value')
    page_result = client.list_import_jobs(request=request)
    for response in page_result:
        print(response)