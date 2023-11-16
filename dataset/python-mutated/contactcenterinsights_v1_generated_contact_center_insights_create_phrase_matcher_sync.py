from google.cloud import contact_center_insights_v1

def sample_create_phrase_matcher():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    phrase_matcher = contact_center_insights_v1.PhraseMatcher()
    phrase_matcher.type_ = 'ANY_OF'
    request = contact_center_insights_v1.CreatePhraseMatcherRequest(parent='parent_value', phrase_matcher=phrase_matcher)
    response = client.create_phrase_matcher(request=request)
    print(response)