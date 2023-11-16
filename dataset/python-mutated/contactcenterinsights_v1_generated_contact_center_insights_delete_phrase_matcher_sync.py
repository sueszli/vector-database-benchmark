from google.cloud import contact_center_insights_v1

def sample_delete_phrase_matcher():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.DeletePhraseMatcherRequest(name='name_value')
    client.delete_phrase_matcher(request=request)