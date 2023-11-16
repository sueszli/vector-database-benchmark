from google.analytics import admin_v1beta

def sample_delete_measurement_protocol_secret():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeleteMeasurementProtocolSecretRequest(name='name_value')
    client.delete_measurement_protocol_secret(request=request)