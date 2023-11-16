from google.cloud import monitoring_dashboard_v1

def sample_delete_dashboard():
    if False:
        return 10
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    request = monitoring_dashboard_v1.DeleteDashboardRequest(name='name_value')
    client.delete_dashboard(request=request)