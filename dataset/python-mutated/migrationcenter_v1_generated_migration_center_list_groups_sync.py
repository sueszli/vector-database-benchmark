from google.cloud import migrationcenter_v1

def sample_list_groups():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ListGroupsRequest(parent='parent_value')
    page_result = client.list_groups(request=request)
    for response in page_result:
        print(response)