from google.cloud import migrationcenter_v1

def sample_list_report_configs():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ListReportConfigsRequest(parent='parent_value')
    page_result = client.list_report_configs(request=request)
    for response in page_result:
        print(response)