from google.analytics import admin_v1beta

def sample_get_measurement_protocol_secret():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.GetMeasurementProtocolSecretRequest(name='name_value')
    response = client.get_measurement_protocol_secret(request=request)
    print(response)