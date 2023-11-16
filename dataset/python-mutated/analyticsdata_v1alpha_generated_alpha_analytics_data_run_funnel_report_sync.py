from google.analytics import data_v1alpha

def sample_run_funnel_report():
    if False:
        print('Hello World!')
    client = data_v1alpha.AlphaAnalyticsDataClient()
    request = data_v1alpha.RunFunnelReportRequest()
    response = client.run_funnel_report(request=request)
    print(response)