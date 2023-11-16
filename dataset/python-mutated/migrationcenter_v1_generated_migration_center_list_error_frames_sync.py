from google.cloud import migrationcenter_v1

def sample_list_error_frames():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ListErrorFramesRequest(parent='parent_value')
    page_result = client.list_error_frames(request=request)
    for response in page_result:
        print(response)