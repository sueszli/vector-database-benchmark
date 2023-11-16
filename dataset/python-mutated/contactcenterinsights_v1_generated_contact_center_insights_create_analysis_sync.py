from google.cloud import contact_center_insights_v1

def sample_create_analysis():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.CreateAnalysisRequest(parent='parent_value')
    operation = client.create_analysis(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)