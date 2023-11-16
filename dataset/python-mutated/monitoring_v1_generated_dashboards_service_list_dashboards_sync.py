from google.cloud import monitoring_dashboard_v1

def sample_list_dashboards():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    request = monitoring_dashboard_v1.ListDashboardsRequest(parent='parent_value')
    page_result = client.list_dashboards(request=request)
    for response in page_result:
        print(response)