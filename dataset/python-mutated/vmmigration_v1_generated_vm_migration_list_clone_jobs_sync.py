from google.cloud import vmmigration_v1

def sample_list_clone_jobs():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListCloneJobsRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_clone_jobs(request=request)
    for response in page_result:
        print(response)