from google.cloud import contact_center_insights_v1

def sample_list_issue_models():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.ListIssueModelsRequest(parent='parent_value')
    response = client.list_issue_models(request=request)
    print(response)