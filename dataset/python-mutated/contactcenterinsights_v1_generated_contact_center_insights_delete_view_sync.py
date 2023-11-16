from google.cloud import contact_center_insights_v1

def sample_delete_view():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.DeleteViewRequest(name='name_value')
    client.delete_view(request=request)