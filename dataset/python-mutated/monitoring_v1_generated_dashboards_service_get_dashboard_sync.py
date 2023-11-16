from google.cloud import monitoring_dashboard_v1

def sample_get_dashboard():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    request = monitoring_dashboard_v1.GetDashboardRequest(name='name_value')
    response = client.get_dashboard(request=request)
    print(response)