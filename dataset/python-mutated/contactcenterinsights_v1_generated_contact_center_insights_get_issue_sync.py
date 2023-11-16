from google.cloud import contact_center_insights_v1

def sample_get_issue():
    if False:
        print('Hello World!')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.GetIssueRequest(name='name_value')
    response = client.get_issue(request=request)
    print(response)