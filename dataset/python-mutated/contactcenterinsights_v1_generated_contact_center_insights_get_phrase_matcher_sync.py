from google.cloud import contact_center_insights_v1

def sample_get_phrase_matcher():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.GetPhraseMatcherRequest(name='name_value')
    response = client.get_phrase_matcher(request=request)
    print(response)