from google.cloud import monitoring_metrics_scope_v1

def sample_get_metrics_scope():
    if False:
        i = 10
        return i + 15
    client = monitoring_metrics_scope_v1.MetricsScopesClient()
    request = monitoring_metrics_scope_v1.GetMetricsScopeRequest(name='name_value')
    response = client.get_metrics_scope(request=request)
    print(response)