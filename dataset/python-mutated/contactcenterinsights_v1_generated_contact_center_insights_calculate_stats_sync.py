from google.cloud import contact_center_insights_v1

def sample_calculate_stats():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.CalculateStatsRequest(location='location_value')
    response = client.calculate_stats(request=request)
    print(response)