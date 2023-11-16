from google.analytics import admin_v1beta

def sample_run_access_report():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.RunAccessReportRequest()
    response = client.run_access_report(request=request)
    print(response)