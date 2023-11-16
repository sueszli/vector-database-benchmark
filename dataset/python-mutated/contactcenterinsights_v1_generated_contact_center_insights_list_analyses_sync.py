from google.cloud import contact_center_insights_v1

def sample_list_analyses():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.ListAnalysesRequest(parent='parent_value')
    page_result = client.list_analyses(request=request)
    for response in page_result:
        print(response)