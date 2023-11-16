from google.cloud import webrisk_v1

def sample_search_uris():
    if False:
        for i in range(10):
            print('nop')
    client = webrisk_v1.WebRiskServiceClient()
    request = webrisk_v1.SearchUrisRequest(uri='uri_value', threat_types=['SOCIAL_ENGINEERING_EXTENDED_COVERAGE'])
    response = client.search_uris(request=request)
    print(response)