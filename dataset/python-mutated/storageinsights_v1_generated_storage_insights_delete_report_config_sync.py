from google.cloud import storageinsights_v1

def sample_delete_report_config():
    if False:
        return 10
    client = storageinsights_v1.StorageInsightsClient()
    request = storageinsights_v1.DeleteReportConfigRequest(name='name_value')
    client.delete_report_config(request=request)