from google.cloud import contact_center_insights_v1

def sample_bulk_analyze_conversations():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.BulkAnalyzeConversationsRequest(parent='parent_value', filter='filter_value', analysis_percentage=0.20170000000000002)
    operation = client.bulk_analyze_conversations(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)