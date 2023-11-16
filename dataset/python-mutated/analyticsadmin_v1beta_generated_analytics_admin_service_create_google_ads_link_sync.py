from google.analytics import admin_v1beta

def sample_create_google_ads_link():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.CreateGoogleAdsLinkRequest(parent='parent_value')
    response = client.create_google_ads_link(request=request)
    print(response)