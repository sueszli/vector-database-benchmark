from google.cloud import storageinsights_v1

def sample_list_report_details():
    if False:
        print('Hello World!')
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.ListReportDetailsRequest(parent='parent_value')
    page_result = client.list_report_details(request=request)
    for response in page_result:
        print(response)