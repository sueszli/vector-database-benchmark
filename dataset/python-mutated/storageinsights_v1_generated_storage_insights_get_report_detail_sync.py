from google.cloud import storageinsights_v1

def sample_get_report_detail():
    if False:
        while True:
            i = 10
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.GetReportDetailRequest(name='name_value')
    response = client.get_report_detail(request=request)
    print(response)