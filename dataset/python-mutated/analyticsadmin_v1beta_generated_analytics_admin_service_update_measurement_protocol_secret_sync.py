from google.analytics import admin_v1beta

def sample_update_measurement_protocol_secret():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    measurement_protocol_secret = admin_v1beta.MeasurementProtocolSecret()
    measurement_protocol_secret.display_name = 'display_name_value'
    request = admin_v1beta.UpdateMeasurementProtocolSecretRequest(measurement_protocol_secret=measurement_protocol_secret)
    response = client.update_measurement_protocol_secret(request=request)
    print(response)