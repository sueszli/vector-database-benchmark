from google.cloud import contact_center_insights_v1

def sample_delete_issue():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.DeleteIssueRequest(name='name_value')
    client.delete_issue(request=request)