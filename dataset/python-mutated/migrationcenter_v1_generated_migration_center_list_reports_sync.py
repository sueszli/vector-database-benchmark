from google.cloud import migrationcenter_v1

def sample_list_reports():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ListReportsRequest(parent='parent_value')
    page_result = client.list_reports(request=request)
    for response in page_result:
        print(response)