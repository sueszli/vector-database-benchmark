from google.analytics import admin_v1beta

def sample_acknowledge_user_data_collection():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.AcknowledgeUserDataCollectionRequest(property='property_value', acknowledgement='acknowledgement_value')
    response = client.acknowledge_user_data_collection(request=request)
    print(response)