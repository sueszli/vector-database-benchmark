from google.cloud import monitoring_metrics_scope_v1

def sample_list_metrics_scopes_by_monitored_project():
    if False:
        while True:
            i = 10
    client = monitoring_metrics_scope_v1.MetricsScopesClient()
    request = monitoring_metrics_scope_v1.ListMetricsScopesByMonitoredProjectRequest(monitored_resource_container='monitored_resource_container_value')
    response = client.list_metrics_scopes_by_monitored_project(request=request)
    print(response)