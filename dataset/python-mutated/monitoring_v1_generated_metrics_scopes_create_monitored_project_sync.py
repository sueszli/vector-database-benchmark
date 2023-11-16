from google.cloud import monitoring_metrics_scope_v1

def sample_create_monitored_project():
    if False:
        return 10
    client = monitoring_metrics_scope_v1.MetricsScopesClient()
    request = monitoring_metrics_scope_v1.CreateMonitoredProjectRequest(parent='parent_value')
    operation = client.create_monitored_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)