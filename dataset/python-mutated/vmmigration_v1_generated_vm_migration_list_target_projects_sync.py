from google.cloud import vmmigration_v1

def sample_list_target_projects():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListTargetProjectsRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_target_projects(request=request)
    for response in page_result:
        print(response)