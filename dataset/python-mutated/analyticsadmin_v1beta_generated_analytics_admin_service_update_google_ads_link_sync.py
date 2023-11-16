from google.analytics import admin_v1beta

def sample_update_google_ads_link():
    if False:
        print('Hello World!')
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.UpdateGoogleAdsLinkRequest()
    response = client.update_google_ads_link(request=request)
    print(response)