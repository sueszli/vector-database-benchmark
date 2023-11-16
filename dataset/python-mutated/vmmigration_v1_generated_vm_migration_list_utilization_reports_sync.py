from google.cloud import vmmigration_v1

def sample_list_utilization_reports():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListUtilizationReportsRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_utilization_reports(request=request)
    for response in page_result:
        print(response)