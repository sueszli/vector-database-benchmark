from google.cloud import contact_center_insights_v1

def sample_delete_analysis():
    if False:
        print('Hello World!')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.DeleteAnalysisRequest(name='name_value')
    client.delete_analysis(request=request)