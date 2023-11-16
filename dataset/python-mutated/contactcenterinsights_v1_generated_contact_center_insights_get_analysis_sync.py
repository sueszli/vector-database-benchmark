from google.cloud import contact_center_insights_v1

def sample_get_analysis():
    if False:
        return 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.GetAnalysisRequest(name='name_value')
    response = client.get_analysis(request=request)
    print(response)