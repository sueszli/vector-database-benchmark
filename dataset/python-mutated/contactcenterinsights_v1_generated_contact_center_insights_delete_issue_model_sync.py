from google.cloud import contact_center_insights_v1

def sample_delete_issue_model():
    if False:
        return 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.DeleteIssueModelRequest(name='name_value')
    operation = client.delete_issue_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)