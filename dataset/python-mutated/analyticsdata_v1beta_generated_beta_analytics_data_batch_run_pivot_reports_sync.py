from google.analytics import data_v1beta

def sample_batch_run_pivot_reports():
    if False:
        for i in range(10):
            print('nop')
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.BatchRunPivotReportsRequest()
    response = client.batch_run_pivot_reports(request=request)
    print(response)