from google.cloud import contact_center_insights_v1

def sample_update_issue():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.UpdateIssueRequest()
    response = client.update_issue(request=request)
    print(response)