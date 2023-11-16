from google.analytics import admin_v1beta

def sample_create_property():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    property = admin_v1beta.Property()
    property.display_name = 'display_name_value'
    property.time_zone = 'time_zone_value'
    request = admin_v1beta.CreatePropertyRequest(property=property)
    response = client.create_property(request=request)
    print(response)