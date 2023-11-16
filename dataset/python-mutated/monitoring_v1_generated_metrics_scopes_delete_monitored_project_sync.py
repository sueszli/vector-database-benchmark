from google.cloud import monitoring_metrics_scope_v1

def sample_delete_monitored_project():
    if False:
        while True:
            i = 10
    client = monitoring_metrics_scope_v1.MetricsScopesClient()
    request = monitoring_metrics_scope_v1.DeleteMonitoredProjectRequest(name='name_value')
    operation = client.delete_monitored_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)