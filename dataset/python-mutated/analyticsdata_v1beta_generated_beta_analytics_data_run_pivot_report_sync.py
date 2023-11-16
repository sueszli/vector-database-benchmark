from google.analytics import data_v1beta

def sample_run_pivot_report():
    if False:
        return 10
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.RunPivotReportRequest()
    response = client.run_pivot_report(request=request)
    print(response)