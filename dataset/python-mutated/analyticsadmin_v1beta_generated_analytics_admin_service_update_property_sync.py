from google.analytics import admin_v1beta

def sample_update_property():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    property = admin_v1beta.Property()
    property.display_name = 'display_name_value'
    property.time_zone = 'time_zone_value'
    request = admin_v1beta.UpdatePropertyRequest(property=property)
    response = client.update_property(request=request)
    print(response)