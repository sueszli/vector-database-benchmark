from google.cloud import contact_center_insights_v1

def sample_undeploy_issue_model():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.UndeployIssueModelRequest(name='name_value')
    operation = client.undeploy_issue_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)