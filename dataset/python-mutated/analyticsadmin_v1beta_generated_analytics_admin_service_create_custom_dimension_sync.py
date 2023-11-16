from google.analytics import admin_v1beta

def sample_create_custom_dimension():
    if False:
        return 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    custom_dimension = admin_v1beta.CustomDimension()
    custom_dimension.parameter_name = 'parameter_name_value'
    custom_dimension.display_name = 'display_name_value'
    custom_dimension.scope = 'ITEM'
    request = admin_v1beta.CreateCustomDimensionRequest(parent='parent_value', custom_dimension=custom_dimension)
    response = client.create_custom_dimension(request=request)
    print(response)