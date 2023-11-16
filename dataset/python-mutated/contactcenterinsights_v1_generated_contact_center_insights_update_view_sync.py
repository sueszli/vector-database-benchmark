from google.cloud import contact_center_insights_v1

def sample_update_view():
    if False:
        return 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.UpdateViewRequest()
    response = client.update_view(request=request)
    print(response)