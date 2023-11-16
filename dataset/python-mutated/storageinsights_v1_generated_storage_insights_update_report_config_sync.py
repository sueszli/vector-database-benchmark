from google.cloud import storageinsights_v1

def sample_update_report_config():
    if False:
        i = 10
        return i + 15
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.UpdateReportConfigRequest()
    response = client.update_report_config(request=request)
    print(response)