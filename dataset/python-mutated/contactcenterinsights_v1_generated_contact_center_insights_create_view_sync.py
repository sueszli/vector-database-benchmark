from google.cloud import contact_center_insights_v1

def sample_create_view():
    if False:
        print('Hello World!')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.CreateViewRequest(parent='parent_value')
    response = client.create_view(request=request)
    print(response)