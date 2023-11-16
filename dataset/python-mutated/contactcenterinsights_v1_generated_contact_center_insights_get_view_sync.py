from google.cloud import contact_center_insights_v1

def sample_get_view():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.GetViewRequest(name='name_value')
    response = client.get_view(request=request)
    print(response)