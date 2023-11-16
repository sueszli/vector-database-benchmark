from google.analytics import data_v1beta

def sample_run_report():
    if False:
        i = 10
        return i + 15
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.RunReportRequest()
    response = client.run_report(request=request)
    print(response)