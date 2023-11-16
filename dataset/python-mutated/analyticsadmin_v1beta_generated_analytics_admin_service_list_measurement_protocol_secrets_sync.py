from google.analytics import admin_v1beta

def sample_list_measurement_protocol_secrets():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ListMeasurementProtocolSecretsRequest(parent='parent_value')
    page_result = client.list_measurement_protocol_secrets(request=request)
    for response in page_result:
        print(response)