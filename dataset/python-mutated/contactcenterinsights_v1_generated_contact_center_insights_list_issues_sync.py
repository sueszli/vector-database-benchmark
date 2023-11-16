from google.cloud import contact_center_insights_v1

def sample_list_issues():
    if False:
        return 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.ListIssuesRequest(parent='parent_value')
    response = client.list_issues(request=request)
    print(response)