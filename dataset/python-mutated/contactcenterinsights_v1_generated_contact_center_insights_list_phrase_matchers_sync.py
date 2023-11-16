from google.cloud import contact_center_insights_v1

def sample_list_phrase_matchers():
    if False:
        print('Hello World!')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.ListPhraseMatchersRequest(parent='parent_value')
    page_result = client.list_phrase_matchers(request=request)
    for response in page_result:
        print(response)