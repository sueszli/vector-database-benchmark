from google.analytics import data_v1beta

def sample_run_realtime_report():
    if False:
        while True:
            i = 10
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.RunRealtimeReportRequest()
    response = client.run_realtime_report(request=request)
    print(response)