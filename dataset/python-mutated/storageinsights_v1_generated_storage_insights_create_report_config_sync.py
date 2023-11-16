from google.cloud import storageinsights_v1

def sample_create_report_config():
    if False:
        i = 10
        return i + 15
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.CreateReportConfigRequest(parent='parent_value')
    response = client.create_report_config(request=request)
    print(response)