from google.analytics import data_v1beta

def sample_check_compatibility():
    if False:
        i = 10
        return i + 15
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.CheckCompatibilityRequest()
    response = client.check_compatibility(request=request)
    print(response)