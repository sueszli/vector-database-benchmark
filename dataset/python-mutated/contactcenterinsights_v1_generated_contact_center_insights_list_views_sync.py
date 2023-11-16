from google.cloud import contact_center_insights_v1

def sample_list_views():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.ListViewsRequest(parent='parent_value')
    page_result = client.list_views(request=request)
    for response in page_result:
        print(response)