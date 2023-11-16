from google.cloud import contact_center_insights_v1

def sample_update_phrase_matcher():
    if False:
        i = 10
        return i + 15
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    phrase_matcher = contact_center_insights_v1.PhraseMatcher()
    phrase_matcher.type_ = 'ANY_OF'
    request = contact_center_insights_v1.UpdatePhraseMatcherRequest(phrase_matcher=phrase_matcher)
    response = client.update_phrase_matcher(request=request)
    print(response)