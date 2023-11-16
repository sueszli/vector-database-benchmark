from google.analytics import data_v1beta

def sample_batch_run_reports():
    if False:
        while True:
            i = 10
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.BatchRunReportsRequest()
    response = client.batch_run_reports(request=request)
    print(response)