from google.cloud import storageinsights_v1

def sample_list_report_configs():
    if False:
        for i in range(10):
            print('nop')
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.ListReportConfigsRequest(parent='parent_value')
    page_result = client.list_report_configs(request=request)
    for response in page_result:
        print(response)