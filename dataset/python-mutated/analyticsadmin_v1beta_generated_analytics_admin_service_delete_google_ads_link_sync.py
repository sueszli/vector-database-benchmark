from google.analytics import admin_v1beta

def sample_delete_google_ads_link():
    if False:
        while True:
            i = 10
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.DeleteGoogleAdsLinkRequest(name='name_value')
    client.delete_google_ads_link(request=request)