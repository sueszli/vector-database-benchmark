from google.cloud import monitoring_dashboard_v1

def sample_update_dashboard():
    if False:
        return 10
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    dashboard = monitoring_dashboard_v1.Dashboard()
    dashboard.display_name = 'display_name_value'
    request = monitoring_dashboard_v1.UpdateDashboardRequest(dashboard=dashboard)
    response = client.update_dashboard(request=request)
    print(response)