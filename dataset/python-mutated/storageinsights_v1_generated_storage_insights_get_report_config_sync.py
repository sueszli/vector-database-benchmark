from google.cloud import storageinsights_v1

def sample_get_report_config():
    if False:
        i = 10
        return i + 15
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.GetReportConfigRequest(name='name_value')
    response = client.get_report_config(request=request)
    print(response)