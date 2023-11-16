from google.cloud import monitoring_dashboard_v1

def sample_create_dashboard():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    dashboard = monitoring_dashboard_v1.Dashboard()
    dashboard.display_name = 'display_name_value'
    request = monitoring_dashboard_v1.CreateDashboardRequest(parent='parent_value', dashboard=dashboard)
    response = client.create_dashboard(request=request)
    print(response)