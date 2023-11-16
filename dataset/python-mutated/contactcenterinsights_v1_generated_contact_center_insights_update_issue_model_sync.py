from google.cloud import contact_center_insights_v1

def sample_update_issue_model():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.UpdateIssueModelRequest()
    response = client.update_issue_model(request=request)
    print(response)