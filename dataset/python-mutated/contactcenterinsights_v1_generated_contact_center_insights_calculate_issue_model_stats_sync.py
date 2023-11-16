from google.cloud import contact_center_insights_v1

def sample_calculate_issue_model_stats():
    if False:
        return 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.CalculateIssueModelStatsRequest(issue_model='issue_model_value')
    response = client.calculate_issue_model_stats(request=request)
    print(response)