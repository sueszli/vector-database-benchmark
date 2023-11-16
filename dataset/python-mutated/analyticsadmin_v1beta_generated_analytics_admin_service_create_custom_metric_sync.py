from google.analytics import admin_v1beta

def sample_create_custom_metric():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    custom_metric = admin_v1beta.CustomMetric()
    custom_metric.parameter_name = 'parameter_name_value'
    custom_metric.display_name = 'display_name_value'
    custom_metric.measurement_unit = 'HOURS'
    custom_metric.scope = 'EVENT'
    request = admin_v1beta.CreateCustomMetricRequest(parent='parent_value', custom_metric=custom_metric)
    response = client.create_custom_metric(request=request)
    print(response)