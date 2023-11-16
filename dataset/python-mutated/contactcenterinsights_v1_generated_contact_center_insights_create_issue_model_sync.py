from google.cloud import contact_center_insights_v1

def sample_create_issue_model():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.CreateIssueModelRequest(parent='parent_value')
    operation = client.create_issue_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)